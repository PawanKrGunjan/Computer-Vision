import os
import shutil
import cv2
from tkinter import Tk, Frame, Canvas, Button, Label, Checkbutton, Scrollbar, IntVar, LabelFrame, Y, VERTICAL, RIGHT, LEFT, BOTH, DISABLED, NORMAL,Radiobutton, StringVar
from PIL import Image, ImageTk
from ultralytics import YOLO

class MockBox:
    def __init__(self, xyxy, cls, conf=1.0):
        self.xyxy = [xyxy]   # list with one element (coordinates)
        self.cls = cls       # class id as int
        self.conf = conf     # confidence (float)

class MockResults:
    def __init__(self, boxes):
        self.boxes = boxes  # list of MockBox


class LabelApp:
    def __init__(self, root, image_paths, model, class_colors):
        self.root = root
        self.root.title("YOLO Auto Labeling")
        self.image_paths = image_paths
        self.model = model
        self.classes=self.model.names
        self.class_colors=class_colors
        self.index = 0
        self.correct_count = 0
        self.incorrect_count = 0
        self.drawing_mode = False

        self.main_frame = Frame(root)
        self.main_frame.pack(fill=BOTH, expand=True)

        # Left: Image canvas
        self.left_panel = Frame(self.main_frame)
        self.left_panel.pack(side="left", fill=BOTH, expand=True)

        self.canvas_width = 1080
        self.canvas_height = 720
        self.canvas = Canvas(self.left_panel, width=self.canvas_width, height=self.canvas_height, bg="gray")
        self.canvas.pack(fill=BOTH, expand=True, padx=10, pady=10)
        # Coordinate label
        self.coord_label = Label(self.left_panel, text="")
        self.coord_label.pack()

        # Right panel: slimmer and styled
        self.right_panel = Frame(self.main_frame, width=160, bg="#f0f0f0")
        self.right_panel.pack(side="right", fill=Y, padx=5, pady=5)
        self.right_panel.pack_propagate(False)  # Prevent auto-resize

        # --- Predictions section ---
        pred_frame = LabelFrame(self.right_panel, text="🔍 Predictions", font=("Arial", 9, "bold"), bg="#f8f8ff")
        pred_frame.pack(fill="both", expand=True, padx=5, pady=(5, 3))

        self.pred_checkbox_frame = Frame(pred_frame, bg="#f8f8ff")
        self.pred_checkbox_frame.pack(fill="both", expand=True)

        self.pred_scrollbar = Scrollbar(self.pred_checkbox_frame, orient=VERTICAL)
        self.pred_scrollbar.pack(side=RIGHT, fill=Y)

        self.pred_check_container = Canvas(self.pred_checkbox_frame, yscrollcommand=self.pred_scrollbar.set, bg="#f8f8ff", highlightthickness=0)
        self.pred_check_container.pack(side=LEFT, fill="both", expand=True)
        self.pred_scrollbar.config(command=self.pred_check_container.yview)

        self.pred_inner_frame = Frame(self.pred_check_container, bg="#f8f8ff")
        self.pred_check_container.create_window((0, 0), window=self.pred_inner_frame, anchor="nw")
        self.pred_inner_frame.bind("<Configure>", lambda e: self.pred_check_container.configure(scrollregion=self.pred_check_container.bbox("all")))

        # --- Manual Edit section ---
        edit_frame = LabelFrame(self.right_panel, text="✏️ Manual Edit", font=("Arial", 9, "bold"), bg="#fefefe")
        edit_frame.pack(fill="both", padx=5, pady=(3, 5))

        self.reject_draw_btn = Button(edit_frame, text="🖍️ Reject & Draw", command=self.reject_and_draw,
                                    bg="purple", fg="white", font=("Arial", 9), width=18, height=1)
        self.reject_draw_btn.pack(pady=(5, 2))

        self.class_var_frame = Frame(edit_frame, bg="#fefefe")
        self.class_var_frame.pack(pady=2)

        self.class_selection_var = StringVar()
        self.class_selection_var.set("")  # No class selected initially

        for i, cls_name in self.classes.items():
            rb = Radiobutton(
                self.class_var_frame, text=cls_name,
                variable=self.class_selection_var, value=str(i),
                bg="#fefefe", font=("Arial", 8), anchor="w"
            )
            rb.pack(anchor="w", padx=5)

        # List to show manually drawn boxes (you can customize this to your needs)
        self.manual_boxes_vars = []
        self.manual_boxes_checkbuttons = []

        # Control buttons at the bottom
        self.bottom_controls = Frame(root)
        self.bottom_controls.pack(side="bottom", pady=10)

        self.prev_button = Button(
            self.bottom_controls, text="⏮️ Previous", command=self.load_previous_image,
            bg="pink", fg="white", font=("Arial", 12), width=20, height=2
        )
        self.prev_button.pack(side="left", padx=10, pady=10)

        self.accept_btn = Button(
            self.bottom_controls, text="✅ Accept", command=self.accept,
            bg="green", fg="white", font=("Arial", 12), width=20, height=2
        )
        self.accept_btn.pack(side="left", padx=10, pady=10)

        self.next_btn = Button(
            self.bottom_controls, text="➡️ Next", command=self.next_image,
            state=DISABLED, bg="orange", fg="white", font=("Arial", 12), width=20, height=2
        )
        self.next_btn.pack(side="left", padx=10, pady=10)

        # Label(self.right_panel, text="Select Class:", font=("Arial", 10, "bold")).pack(pady=5)
        self.label = Label(root, text="", font=("Arial", 12))
        self.label.pack(pady=8, side="bottom")

        # Bind canvas events for drawing
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<Motion>", self.on_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Button-3>", self.on_right_click)        

        self.root.bind("<Escape>", self.on_escape)

        self.start_x = self.start_y = None
        self.current_rect = None
        self.boxes = []  # manually drawn boxes (x1, y1, x2, y2, cls)
        self.load_image()

    def load_image(self):
        if self.index >= len(self.image_paths):
            self.label.config(text=f"✅ Labeling Complete! Accuracy: {self.calculate_accuracy():.2f}%")
            return

        self.image_path = self.image_paths[self.index]
        self.image = cv2.imread(self.image_path)
        self.h, self.w = self.image.shape[:2]
        self.drawing_mode = False  # reset drawing mode

        # Clear manual boxes
        self.boxes.clear()
        for cb in self.manual_boxes_checkbuttons:
            cb.destroy()
        self.manual_boxes_vars.clear()
        self.manual_boxes_checkbuttons.clear()

        # Clear predictions
        self.clear_prediction_checkboxes()

        txt_path = os.path.splitext(self.image_path)[0] + ".txt"
        # Decide whether to use model or label file
        if os.path.exists(txt_path): #and hasattr(self, "label_from_file") and self.label_from_file:
            #self.label_from_file = False
            self.results = self.load_labels_as_results(self.image_path)
        else:
            self.results = model(self.image_path)[0]

        # Populate prediction checkboxes or manual label boxes
        self.populate_prediction_checkboxes()

        self.next_btn.config(state=DISABLED)
        self.update_status()
        self.display_image_and_boxes()

    def load_labels_as_results(self, image_path):
        txt_path = os.path.splitext(image_path)[0] + ".txt"
        boxes = []

        if not os.path.exists(txt_path):
            return MockResults([])

        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id, x_center, y_center, w, h = map(float, parts)
                x1 = (x_center - w / 2) * self.w
                y1 = (y_center - h / 2) * self.h
                x2 = (x_center + w / 2) * self.w
                y2 = (y_center + h / 2) * self.h
                box = MockBox([x1, y1, x2, y2], int(cls_id), conf=1.0)
                boxes.append(box)

        return MockResults(boxes)

    def clear_prediction_checkboxes(self):
        # Remove all widgets from prediction inner frame
        for widget in self.pred_inner_frame.winfo_children():
            widget.destroy()
        self.checkbox_vars = []

    def populate_prediction_checkboxes(self):
        if self.drawing_mode:
            # When in drawing mode, do NOT show prediction boxes
            return

        for i, box in enumerate(self.results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls)
            label = self.classes[cls] if cls < len(self.classes) else str(cls)
            var = IntVar(value=1)
            cb = Checkbutton(self.pred_inner_frame, text=label, variable=var, font=("Arial", 12), anchor="w", padx=10, width=20)
            cb.pack(anchor="w")
            cb.config(command=self.display_image_and_boxes)  # add this line
            self.checkbox_vars.append((var, box))

                
    def display_image_and_boxes(self):
        self.canvas.delete("all")

        scale = min(self.canvas_width / self.w, self.canvas_height / self.h)
        disp_w, disp_h = int(self.w * scale), int(self.h * scale)

        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb)
        img_pil = img_pil.resize((disp_w, disp_h), Image.Resampling.LANCZOS)

        self.tk_img = ImageTk.PhotoImage(img_pil)

        self.img_x_offset = (self.canvas_width - disp_w) // 2
        self.img_y_offset = (self.canvas_height - disp_h) // 2

        self.canvas.create_image(self.img_x_offset, self.img_y_offset, anchor="nw", image=self.tk_img)

        if not self.drawing_mode:
            for var, box in self.checkbox_vars:
                if var.get() == 1:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1_s = int(x1 * scale) + self.img_x_offset
                    y1_s = int(y1 * scale) + self.img_y_offset
                    x2_s = int(x2 * scale) + self.img_x_offset
                    y2_s = int(y2 * scale) + self.img_y_offset

                    cls = int(box.cls)
                    label = self.classes[cls]
                    color = self.class_colors.get(cls, (0, 0, 0))
                    color_hex = '#%02x%02x%02x' % color

                    self.canvas.create_rectangle(x1_s, y1_s, x2_s, y2_s,
                                                outline=color_hex,
                                                width=2,
                                                fill=color_hex,
                                                stipple='gray25')
                    self.canvas.create_text(x1_s + 3, y1_s + 10,
                                            anchor="nw",
                                            text=label,
                                            fill=color_hex,
                                            font=("Arial", 10, "bold"))

        # Draw manual boxes always
        for (x1, y1, x2, y2, cls) in self.boxes:
            color = self.class_colors.get(cls, (255, 0, 0))
            color_hex = '#%02x%02x%02x' % color
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color_hex, width=2)
            self.canvas.create_text(x1 + 5, y1 + 10, anchor="nw", text=self.classes[cls], fill=color_hex, font=("Arial", 10, "bold"))

    def accept(self):
        # Save annotations (manual or predicted)
        self.save_labels()

        if self.drawing_mode:
            self.correct_count -= 1  # Optional: adjust depending on your logic
            self.drawing_mode = False
        else:
            self.correct_count += 1

        self.next_btn.config(state=NORMAL)
        self.label.config(text="✅ Accepted. Click ➡️ Next to continue.")

    def reject_and_draw(self):
        self.incorrect_count += 1
        self.drawing_mode = True
        self.label.config(text="✏️ Drawing Mode Activated. Draw boxes and click Accept.")
        self.results = []  # clear predictions
        self.boxes.clear()
        self.next_btn.config(state=DISABLED)

        # Clear prediction checkboxes from UI and disable them
        self.clear_prediction_checkboxes()
        self.display_image_and_boxes()

    def next_image(self):
        self.index += 1
        self.boxes.clear()
        self.drawing_mode = False
        self.next_btn.config(state=DISABLED)
        self.load_image()

    def load_previous_image(self):
        if self.index > 0:
            self.index -= 1
            #self.label_from_file = True  # <-- NEW flag to skip prediction
            self.load_image()
        else:
            self.label.config(text="🚫 This is the first image.")

    def save_labels(self):
        label_path = os.path.splitext(self.image_path)[0] + ".txt"
        with open(label_path, "w") as f:
            # Save predictions if not in drawing mode
            if not self.drawing_mode:
                for var, box in self.checkbox_vars:
                    if var.get() == 1:
                        x1, y1, x2, y2 = box.xyxy[0]
                        cls = int(box.cls)
                        cx = ((x1 + x2) / 2) / self.w
                        cy = ((y1 + y2) / 2) / self.h
                        bw = (x2 - x1) / self.w
                        bh = (y2 - y1) / self.h
                        f.write(f"{cls} {cx} {cy} {bw} {bh}\n")

            # Save manual boxes scaled back to original image coords
            scale = min(self.canvas_width / self.w, self.canvas_height / self.h)
            for x1, y1, x2, y2, cls in self.boxes:
                # convert canvas coords to original image coords
                x1_img = (x1 - self.img_x_offset) / scale
                y1_img = (y1 - self.img_y_offset) / scale
                x2_img = (x2 - self.img_x_offset) / scale
                y2_img = (y2 - self.img_y_offset) / scale

                cx = ((x1_img + x2_img) / 2) / self.w
                cy = ((y1_img + y2_img) / 2) / self.h
                bw = abs(x2_img - x1_img) / self.w
                bh = abs(y2_img - y1_img) / self.h
                f.write(f"{cls} {cx} {cy} {bw} {bh}\n")

        print(f"Labeled: {label_path}")

    def update_status(self):
        total = self.correct_count + self.incorrect_count
        accuracy = self.calculate_accuracy()
        self.label.config(
            text=f"Image {self.index + 1}/{len(self.image_paths)}: {os.path.basename(self.image_path)} "
                f"| ✅ Correct: {self.correct_count} | ❌ Incorrect: {self.incorrect_count} | 🎯 Accuracy: {accuracy:.2f}%"
        )

    def calculate_accuracy(self):
        total = self.correct_count + self.incorrect_count
        return (self.correct_count / total * 100) if total > 0 else 0

    def on_click(self, event):
        if self.drawing_mode:
            # Only allow drawing inside image displayed area
            if not (self.img_x_offset <= event.x <= self.img_x_offset + int(self.w * min(self.canvas_width / self.w, self.canvas_height / self.h)) and
                    self.img_y_offset <= event.y <= self.img_y_offset + int(self.h * min(self.canvas_width / self.w, self.canvas_height / self.h))):
                return
            self.start_x, self.start_y = event.x, event.y
            self.current_rect = self.canvas.create_rectangle(self.start_x, self.start_y, event.x, event.y, outline="red", width=2)

    def on_drag(self, event):
        if self.drawing_mode and self.current_rect:
            self.canvas.coords(self.current_rect, self.start_x, self.start_y, event.x, event.y)

    def on_move(self, event):
        # Update coordinates label
        self.coord_label.config(text=f"X: {event.x}, Y: {event.y}")

        # Remove previous lines
        self.canvas.delete("guide")

        # Draw new guide lines with increased thickness
        self.canvas.create_line(
            event.x, 0, event.x, self.h,
            dash=(2, 2),
            fill='gray',
            width=2,
            tags="guide"
        )
        self.canvas.create_line(
            0, event.y, self.w, event.y,
            dash=(2, 2),
            fill='gray',
            width=2,
            tags="guide"
        )

    def on_release(self, event):
        if self.drawing_mode and self.current_rect:
            coords = self.canvas.coords(self.current_rect)
            if len(coords) == 4:
                x1, y1, x2, y2 = coords
                selected_class = self.class_selection_var.get()
                if selected_class != "":
                    cls_index = int(selected_class)
                    self.boxes.append((x1, y1, x2, y2, cls_index))
                else:
                    self.label.config(text="⚠️ Please select a class before drawing boxes.")
            else:
                print("⚠️ Rectangle has invalid or incomplete coordinates, skipping.")
            self.canvas.delete(self.current_rect)
            self.current_rect = None
            self.display_image_and_boxes()

    def on_right_click(self, event):
        # Delete manual boxes on right click inside box
        to_delete = None
        for i, (x1, y1, x2, y2, cls) in enumerate(self.boxes):
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                to_delete = i
                break
        if to_delete is not None:
            del self.boxes[to_delete]
            self.display_image_and_boxes()

    def on_escape(self, event):
        if self.drawing_mode:
            # Clear current manual boxes but stay in drawing mode so user can redraw
            self.boxes.clear()
            self.display_image_and_boxes()
            self.label.config(text="✏️ Drawing reset. Draw boxes and click Accept.")
        else:
            self.root.quit()
        


# === MAIN EXECUTION ===
if __name__ == "__main__":
    MODEL_PATH = r'./ATCC_MODEL/atcc/weights/best.pt'
    model = YOLO(MODEL_PATH)
    #NAMES = ['2 Wheelers', '3 Wheelers', '4 Wheelers', 'LCV', 'Bus', 'Truck', 'Tractor', 'HCM']
    # Define your class names here or load from file
    cls_name = model.names
    print(cls_name)
    cls_colors = {
        0: (255, 0, 0),       # Red
        1: (255, 165, 0),     # Orange
        2: (0, 255, 0),       # Green
        3: (0, 255, 255),     # Cyan
        4: (0, 0, 255),       # Blue
        5: (128, 0, 128),     # Purple
        6: (255, 255, 0),     # Yellow
        7: (0, 128, 128)      # Teal
    }

    image_dir = r'./ATCC_LABEL_NEW/2W'

    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    root = Tk()
    app = LabelApp(root, sorted(image_files), model, cls_colors)
    #app = LabelApp(root, sorted(image_files))
    root.mainloop()
