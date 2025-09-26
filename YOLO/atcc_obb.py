import os
import json
import cv2
import time
from collections import defaultdict, deque
from datetime import datetime
import torch
import numpy as np
from urllib.parse import quote
from ultralytics import YOLO
import traceback
from LaneDrawer import LaneDrawer
import logging
import threading  # kept only for threaded_save
from queue import Full, Empty
import multiprocessing as mp

logger = logging.getLogger("ATCC")


class ATCC:
    def __init__(self,
                 model_path: str = "atcc_obb_876_openvino_model",
                 save_dir: str = 'ATCC_IMAGES',
                 display: bool = False):
        self.model = self.load_model(model_path)
        self.save_dir = save_dir
        # counted_ids should be a deque for membership tests and append
        self.counted_ids = deque(maxlen=500)
        self.max_history = 20
        self.position_history = defaultdict(
            self._deque_factory
        )
        self.vehicle_counts = defaultdict(
            self._int_dict_factory
        )

        self.streaming = False
        self.frame = None
        self.display_enabled = display

        # Realtime counters (per frame)
        self.live_counts = {}
        self.vehicle_stop_counts = {}
        self.stop_threshold_secs = 3  # Tunable

        # THREAD-SAFETY
        self.lock = threading.Lock()

    @staticmethod
    def _deque_factory():
        return deque(maxlen=20)

    @staticmethod
    def _int_dict_factory():
        return defaultdict(int)

    def load_model(self, model_path: str):
        logger.info(f"Loading model from {model_path}")
        try:
            if model_path.endswith(".pt"):
                model = YOLO(model_path, task='track')
                export_result = model.export(format='openvino', dynamic=True)
                model = YOLO(str(export_result), task='obb')
            elif os.path.isdir(model_path):
                model = YOLO(model_path, task='obb')
            else:
                raise FileNotFoundError(f"Invalid model path: {model_path}")
            logger.info(f"Model successfully loaded on {model.device}")
            return model
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise

    def _counts_to_plain_dict(self, vehicle_counts):
        """
        Convert nested defaultdicts to plain dicts suitable for pickling/queue.
        """
        return {lane: dict(classes) for lane, classes in vehicle_counts.items()}

    def get_color_for_class(self, class_id: int):
        class_colors = {
            0: (102, 102, 255),
            1: (102, 178, 255),
            2: (153, 255, 153),
            3: (255, 255, 153),
            4: (255, 204, 153),
            5: (255, 153, 204),
            6: (153, 255, 255),
            7: (225, 230, 179)
        }
        return class_colors.get(class_id, (200, 200, 200))

    def load_lane_region(self, camera_id, rtsp_url):
        lane_json_file = f"lanes_{camera_id.replace('.', '_')}.json"
        if os.path.exists(lane_json_file):
            logger.info(f"Loading lanes from: {lane_json_file}")
            with open(lane_json_file, 'r') as f:
                return json.load(f)
        else:
            drawer = LaneDrawer(rtsp_url)
            drawer.capture_frame()
            drawer.draw_lanes()
            lane_dict = drawer.get_lanes()
            logger.warning(f"No lane JSON found for {camera_id}")
            logger.info(f"Saving lane definitions to: {lane_json_file}")
            with open(lane_json_file, 'w') as f:
                json.dump(lane_dict, f, indent=4)
            return lane_dict

    def save_frame(self, class_name, camera_id):
        os.makedirs(self.save_dir, exist_ok=True)
        camera_dir = os.path.join(self.save_dir, camera_id)
        os.makedirs(camera_dir, exist_ok=True)
        today = datetime.now()
        class_dir = os.path.join(camera_dir, today.strftime('%Y%m%d'), class_name)
        os.makedirs(class_dir, exist_ok=True)
        filename = os.path.join(class_dir, today.strftime('%H_%M_%S.%f') + '.jpg')
        return filename

    def is_vehicle_stopped(self, track_id, cx, cy, current_time):
        history = self.position_history[track_id]
        history.append((cx, cy, current_time))

        filtered = [p for p in history if (current_time - p[2]).total_seconds() <= self.stop_threshold_secs]
        if len(filtered) < 2:
            return False

        x_vals, y_vals = zip(*[(x, y) for x, y, _ in filtered])
        return max(x_vals) - min(x_vals) < 10 and max(y_vals) - min(y_vals) < 10

    def threaded_save(self, img_bytes, filename):
        def _save():
            try:
                with open(filename, 'wb') as f:
                    f.write(img_bytes)
            except Exception as e:
                logger.error(f"Error saving image: {e}")
        threading.Thread(target=_save, daemon=True).start()

    def start_stream(self, camera_id, rtsp_url, frame_proxy=None, counts_proxy=None):
        # NOTE: if you are creating ATCC() in the parent process and then
        # spawning a child Process with a bound method, pickling will attempt
        # to serialize the whole ATCC object. For robust multiprocessing,
        # consider creating the ATCC instance inside the worker process.
        self.stop_stream()
        self.streaming = True
        rtsp_url = f'192.168.7.151.mp4'  # For testing only (keep original RTSP replace in production)
        lane_regions = self.load_lane_region(camera_id=camera_id, rtsp_url=rtsp_url)
        self.stream_loop(rtsp_url, camera_id, lane_regions, frame_proxy, counts_proxy)

    def stop_stream(self):
        self.streaming = False
        self.vehicle_counts.clear()
        self.live_counts.clear()
        self.vehicle_stop_counts.clear()
        self.counted_ids.clear()

    def stream_loop(self, rtsp_url, camera_id, lane_regions, frame_proxy, counts_proxy):
        self.vehicle_counts.clear()

        try:
            results = self.model.track(
                rtsp_url,
                conf=0.5,
                iou=0.4,
                stream=True,
                tracker="bytetrack.yaml",
                max_det=25,
                verbose=False
            )
            logger.info(f"Stream started from: {rtsp_url}")

            for result in results:
                if not self.streaming:
                    logger.info("Stream interrupted by stop_stream.")
                    break

                frame = result.orig_img
                timestamp = datetime.now()
                h, w = frame.shape[:2]

                current_live_counts = defaultdict(int)
                current_stop_counts = defaultdict(lambda: defaultdict(int))

                if result.obb and result.obb.xyxyxyxy is not None:
                    obb_tensor = result.obb.xyxyxyxy
                    if isinstance(obb_tensor, torch.Tensor):
                        obb_tensor = obb_tensor.cpu().numpy()

                    ids = result.obb.id
                    ids = ids.cpu().numpy().astype(int) if ids is not None and isinstance(ids, torch.Tensor) else []

                    classes = result.obb.cls
                    classes = classes.cpu().numpy().astype(int) if classes is not None and isinstance(classes, torch.Tensor) else []

                    for obb, track_id, cls_id in zip(obb_tensor, ids, classes):
                        class_name = self.model.names[cls_id]

                        cx = int(np.mean(obb[:, 0]))
                        cy = int(np.mean(obb[:, 1]))

                        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), 2)

                        lane_inside = None
                        for lane_name, points in lane_regions.items():
                            if cv2.pointPolygonTest(np.array(points, np.int32), (cx, cy), False) >= 0:
                                lane_inside = lane_name
                                break

                        if lane_inside is None:
                            continue

                        current_live_counts[lane_inside] += 1

                        # counted_ids is a deque, so membership check works
                        if track_id not in self.counted_ids and cy > 0.6 * h:
                            self.vehicle_counts[lane_inside][class_name] += 1
                            self.counted_ids.append(track_id)

                            filename = self.save_frame(camera_id, class_name)
                            success, encoded_img = cv2.imencode('.jpg', result.orig_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                            if success:
                                self.threaded_save(encoded_img.tobytes(), filename)

                        if self.is_vehicle_stopped(track_id, cx, cy, timestamp):
                            current_stop_counts[lane_inside][class_name] += 1

                        color = self.get_color_for_class(cls_id)
                        points = obb.reshape((-1, 1, 2)).astype(int)
                        cv2.polylines(frame, [points], isClosed=True, color=color, thickness=2)

                for lane_name, points in lane_regions.items():
                    pts = np.array(points, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
                    cx = int(sum(p[0] for p in points) / len(points))
                    cy = int(sum(p[1] for p in points) / len(points))
                    cv2.putText(frame, lane_name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 1)

                if frame_proxy is not None and counts_proxy is not None:
                    # Put counts snapshot as a plain dict (no defaultdicts)
                    counts_snapshot = self._counts_to_plain_dict(self.vehicle_counts)
                    try:
                        counts_proxy.put_nowait(counts_snapshot)
                    except Full:
                        # drop oldest snapshot, then push newest
                        try:
                            _ = counts_proxy.get_nowait()
                            counts_proxy.put_nowait(counts_snapshot)
                        except Exception:
                            pass

                    # Encode current frame to JPEG and put bytes into frame_queue
                    success, encoded_img = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if success:
                        frame_bytes = encoded_img.tobytes()
                        try:
                            frame_proxy.put_nowait(frame_bytes)
                        except Full:
                            # drop the oldest frame then put new
                            try:
                                _ = frame_proxy.get_nowait()
                                frame_proxy.put_nowait(frame_bytes)
                            except Exception:
                                pass
                    else:
                        logger.warning("JPEG encoding failed for frame")
                else:
                    # Local-mode fallback
                    self.live_counts = current_live_counts
                    self.vehicle_stop_counts = current_stop_counts
                    self.frame = frame
                    #logger.error("counts_proxy and frame_proxy not defined")

                if self.display_enabled:
                    resized_frame = cv2.resize(frame, (640, 480))
                    cv2.imshow('ATCC Stream', resized_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("'q' pressed - exiting stream.")
                        break

        except Exception as e:
            logger.error(f"Stream error: {e}")
            logger.error(traceback.format_exc())
        finally:
            logger.info("Stream finished. Cleaning up display.")
            cv2.destroyAllWindows()
            for handler in logger.handlers:
                try:
                    handler.flush()
                except Exception:
                    pass


if __name__ == "__main__":
    def to_dict(d):
        if isinstance(d, defaultdict):
            d = {k: to_dict(v) for k, v in d.items()}
        elif isinstance(d, dict):
            d = {k: to_dict(v) for k, v in d.items()}
        return d

    tracker = ATCC(
        model_path="atcc_obb_876.pt",
        save_dir="ATCC_IMAGES",
        display=True
    )

    camera_id = "Test_00"
    username = 'admin'
    password = 'metro@123'
    camera_ip = '192.168.10.63'
    stream_name = 'cam/realmonitor?channel=1&subtype=0'
    encoded_password = quote(password, safe='')
    rtsp_url = f"rtsp://{username}:{encoded_password}@{camera_ip}:554/{stream_name}"
    rtsp_url = "192.168.7.151.mp4"
    tracker.start_stream(camera_id, rtsp_url)

    try:
        print("Monitoring vehicle data (press Ctrl+C to stop)...")
        while True:
            time.sleep(2)
            print("\nTotal Vehicle Counts (per lane/class):")
            print(json.dumps(to_dict(tracker.vehicle_counts), indent=2))
            print("\nLive Vehicle Counts (currently tracked):")
            print(json.dumps(to_dict(tracker.live_counts), indent=2))
            print("\nStopped Vehicle Counts:")
            print(json.dumps(to_dict(tracker.vehicle_stop_counts), indent=2))

    except KeyboardInterrupt:
        print("Interrupted by user. Stopping...")
        tracker.stop_stream()
    except Exception as e:
        print("Error occurred during stream:")
        print(str(e))
        tracker.stop_stream()
