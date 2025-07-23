
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import cv2
import os
import numpy as np

from ultralytics import YOLO  # use pretrained YOLOv8

# Preprocess and extract embedding


class ScrollableFrame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        
        canvas = tk.Canvas(self)
        scrollbar = tk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        
class MOTAnnotationEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("MC-MOT Annotation Tool")
        self.root.iconbitmap("assets/app_icon.ico")  

        # Enable resizing
        self.root.grid_rowconfigure(5, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        # Variables
        self.vid_dir = ""
        self.ann_dir = ""
        self.out_dir = ""
        self.video_caps = {}
        self.annotations = {}
        self.current_frame_ids = {}
        self.video_paths = {}
        self.video_sizes = {}
        self.canvases = {}
        self.frame_labels = {}  
        self.drawing = False
        self.start_x = self.start_y = 0
        self.rectangles = {}
        self.person_model = YOLO("model/weights/best.pt")  # or yolov8s.pt for more accuracy
        # GUI Elements
        self.create_widgets()

        import json
        # Keyboard bindings for next/prev frame
        self.root.bind('<Left>', lambda e: self.prev_frame())
        self.root.bind('<Right>', lambda e: self.next_frame())
        from torchreid.utils.feature_extractor import FeatureExtractor
        import torch

        self.reid_extractor = FeatureExtractor(
            model_name='osnet_ibn_x1_0',
            model_path='',  # pretrained by default
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.next_global_id = 1
        self.global_threshold = 0.8    # cosine threshold
        self.global_gallery = {}  # global_id -> [features]
        self.global_id_history = {}  # global_id -> list of (cam, frame_id, bbox)
        self.next_global_id = 1
        self.iou_threshold = 0.35
    
    def extract_bbox_feature(self, frame, bbox):
        x, y, w, h = bbox
        crop = frame[y:y+h, x:x+w]
        features = self.reid_extractor([crop])
        return features[0]  # tensor embedding
    
    
    def assign_global_ids(self, cam, frame_id, frame, max_history_frames=5):
        import torch

        for obj in self.annotations[cam].get(frame_id, []):
            feat = self.extract_bbox_feature(frame, obj['bbox'])
            feat_norm = feat / feat.norm()
            assigned = False

            # 1. Search IoU with previous N frames across all cams
            for cam2 in self.annotations:
                for past_fid in range(max(0, frame_id - max_history_frames), frame_id)[::-1]:
                    for prev_obj in self.annotations[cam2].get(past_fid, []):
                        if 'global_id' in prev_obj:
                            iou = self.compute_iou(obj['bbox'], prev_obj['bbox'])
                            if iou > self.iou_threshold:
                                obj['global_id'] = prev_obj['global_id']
                                self.global_gallery[prev_obj['global_id']].append(feat_norm)
                                self.global_id_history[prev_obj['global_id']].append((cam, frame_id, obj['bbox']))
                                assigned = True
                                break
                    if assigned:
                        break
                if assigned:
                    break

            if assigned:
                continue  # Already assigned using motion IoU

            # 2. Search by feature similarity
            best_id, best_sim = None, -1.0
            for gid, feats in self.global_gallery.items():
                sims = torch.stack([feat_norm @ f for f in feats])
                sim = sims.max().item()
                if sim > best_sim:
                    best_sim = sim
                    best_id = gid

            if best_sim >= self.global_threshold:
                obj['global_id'] = best_id
                self.global_gallery[best_id].append(feat_norm)
                self.global_id_history[best_id].append((cam, frame_id, obj['bbox']))
            else:
                # 3. Only now assign new ID
                gid = self.next_global_id
                self.next_global_id += 1
                obj['global_id'] = gid
                self.global_gallery[gid] = [feat_norm]
                self.global_id_history[gid] = [(cam, frame_id, obj['bbox'])]


    def load_last_frame(self):
        import json
        try:
            with open("checkpoint_frame.json", "r") as f:
                self.current_frame_ids = json.load(f)
        except FileNotFoundError:
            pass

    def save_last_frame(self):
        import json
        with open("checkpoint_frame.json", "w") as f:
            json.dump(self.current_frame_ids, f)

        
    def detect_persons_with_yolo(self, frame):
        results = self.person_model(frame)[0]
        detections = []
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls == 0:  # person class
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                w, h = x2 - x1, y2 - y1
                detections.append({'bbox': [x1, y1, w, h]})
        return detections

    def compute_iou(self,box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        xa = max(x1, x2)
        ya = max(y1, y2)
        xb = min(x1 + w1, x2 + w2)
        yb = min(y1 + h1, y2 + h2)
        inter = max(0, xb - xa) * max(0, yb - ya)
        union = w1 * h1 + w2 * h2 - inter
        return inter / union if union > 0 else 0

    def generate_new_id(self, cam):
        # Assign max existing ID + 1
        all_ids = [obj['id'] for f_objs in self.annotations[cam].values() for obj in f_objs]
        return max(all_ids) + 1 if all_ids else 1

    
    def auto_detect_missing(self):
        for cam in self.frames:
            fid = self.current_frame_ids[cam]
            if fid >= len(self.frames[cam]):
                continue
            frame = cv2.imread(self.frames[cam][fid])
            if frame is None:
                continue

            existing = [obj['bbox'] for obj in self.annotations.get(cam, {}).get(fid, [])]
            dets = self.detect_persons_with_yolo(frame)

            for d in dets:
                if not any(self.compute_iou(d['bbox'], ex) > 0.5 for ex in existing):
                    nid = self.generate_new_id(cam)
                    self.annotations[cam].setdefault(fid, []).append({'id': nid, 'bbox': d['bbox']})

            # 🎯 Assign global IDs based on ReID
            self.assign_global_ids(cam, fid, frame)

        self.show_all_frames()

        
    def create_widgets(self):
        # Directory selection
        tk.Label(self.root, text="Videos Folder:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.vid_entry = tk.Entry(self.root, width=50)
        self.vid_entry.grid(row=0, column=1, padx=5, pady=5)
        tk.Button(self.root, text="Browse", command=self.select_vid_dir).grid(row=0, column=2, padx=5, pady=5)

        tk.Label(self.root, text="Annotations Folder:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.ann_entry = tk.Entry(self.root, width=50)
        self.ann_entry.grid(row=1, column=1, padx=5, pady=5)
        tk.Button(self.root, text="Browse", command=self.select_ann_dir).grid(row=1, column=2, padx=5, pady=5)

        tk.Label(self.root, text="Output Folder:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.out_entry = tk.Entry(self.root, width=50)
        self.out_entry.grid(row=2, column=1, padx=5, pady=5)
        tk.Button(self.root, text="Browse", command=self.select_out_dir).grid(row=2, column=2, padx=5, pady=5)

        tk.Button(self.root, text="Load Videos & Annotations", command=self.load_videos_and_annotations).grid(
            row=3, column=1, padx=5, pady=10
        )
        

        # Control buttons
        self.control_frame = tk.Frame(self.root)
        self.control_frame.grid(row=4, column=0, columnspan=3)

        tk.Button(self.control_frame, text="Previous Frame", command=self.prev_frame).grid(row=0, column=0, padx=5)
        tk.Button(self.control_frame, text="Next Frame", command=self.next_frame).grid(row=0, column=1, padx=5)
        tk.Button(self.control_frame, text="Save Annotations", command=self.save_annotations).grid(row=0, column=2, padx=5)
        tk.Button(self.control_frame, text="Auto Detect", command=self.auto_detect_missing).grid(row=0, column=3, padx=5)

        # Frame to hold all canvases
        
        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame = ScrollableFrame(self.root)
        self.canvas_frame.grid(row=5, column=0, columnspan=3, padx=5, pady=5)
        
        
        self.outer_canvas = tk.Canvas(self.root, height=600)  # fixed height for scrollbar effect      
        
        self.outer_canvas.grid(row=5, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)

        self.v_scrollbar = tk.Scrollbar(self.root, orient="vertical", command=self.outer_canvas.yview)
        self.v_scrollbar.grid(row=5, column=3, sticky="ns")

        self.outer_canvas.configure(yscrollcommand=self.v_scrollbar.set)

        self.canvas_frame = tk.Frame(self.outer_canvas)
        self.canvas_frame_id = self.outer_canvas.create_window((0, 0), window=self.canvas_frame, anchor='nw')

        # Bind to update scrollregion when canvas_frame changes size
        def on_frame_configure(event):
            self.outer_canvas.configure(scrollregion=self.outer_canvas.bbox("all"))

        self.canvas_frame.bind("<Configure>", on_frame_configure)

        # Optional: Make canvas_frame expand width with outer_canvas width
        def on_canvas_configure(event):
            self.outer_canvas.itemconfig(self.canvas_frame_id, width=event.width)

        self.outer_canvas.bind("<Configure>", on_canvas_configure)
        
        
        self.root.grid_rowconfigure(5, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        
        
        def _on_mousewheel(event):
            self.outer_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        self.outer_canvas.bind_all("<MouseWheel>", _on_mousewheel)

    def select_vid_dir(self):
        self.vid_dir = filedialog.askdirectory()
        self.vid_entry.delete(0, tk.END)
        self.vid_entry.insert(0, self.vid_dir)

    def select_ann_dir(self):
        self.ann_dir = filedialog.askdirectory()
        self.ann_entry.delete(0, tk.END)
        self.ann_entry.insert(0, self.ann_dir)

    def select_out_dir(self):
        self.out_dir = filedialog.askdirectory()
        self.out_entry.delete(0, tk.END)
        self.out_entry.insert(0, self.out_dir)

    # def load_videos_and_annotations(self):
    #     if not os.path.isdir(self.vid_dir) or not os.path.isdir(self.ann_dir):
    #         messagebox.showerror("Error", "Invalid video or annotation directory.")
    #         return
    import cv2
    import os
    from tkinter import messagebox
    
    
    def extract_frames(self, video_path, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out_path = os.path.join(output_dir, f"frame_{count:05d}.jpg")
            cv2.imwrite(out_path, frame)
            count += 1
        cap.release()
        print(f"Extracted {count} frames from {os.path.basename(video_path)}")
        

    def load_videos_and_annotations(self):
        import cv2
        import os

        if not os.path.isdir(self.vid_dir) or not os.path.isdir(self.ann_dir):
            messagebox.showerror("Error", "Invalid video or annotation directory.")
            return
        if not self.out_dir:
            messagebox.showerror("Error", "Please select an output folder.")
            return

        os.makedirs(self.out_dir, exist_ok=True)

        video_files = [f for f in os.listdir(self.vid_dir) if f.endswith('.mp4')]
        self.video_paths = {}
        self.frames = {}
        self.video_caps = {}
        self.video_sizes = {}

        for vf in video_files:
            base_name = os.path.splitext(vf)[0]
            ann_file = os.path.join(self.ann_dir, f"{base_name}.txt")
            if not os.path.exists(ann_file):
                print(f"Warning: No matching annotation file for {vf}")
                continue

            video_path = os.path.join(self.vid_dir, vf)
            frame_dir = os.path.join(self.vid_dir, f"{base_name}_frames")

            # Use existing frames if they exist
            if not os.path.exists(frame_dir) or not os.listdir(frame_dir):
                print(f"Extracting frames for {base_name}...")
                self.extract_frames(video_path, frame_dir)
            else:
                print(f"Using existing frames for {base_name}...")

            frame_files = sorted([
                os.path.join(frame_dir, f)
                for f in os.listdir(frame_dir)
                if f.endswith(('.jpg', '.png'))
            ])

            if not frame_files:
                print(f"No frames found for {vf}, skipping.")
                continue

            self.video_paths[base_name] = video_path
            self.frames[base_name] = frame_files

            # Get size from first frame
            sample_frame = cv2.imread(frame_files[0])
            h, w = sample_frame.shape[:2]
            self.video_sizes[base_name] = (w, h)

        if not self.video_paths:
            messagebox.showerror("Error", "No matching video and annotation pairs found.")
            return

        self.video_caps = {cam: None for cam in self.video_paths}

        self.annotations = {
            cam: load_mot_annotations(os.path.join(self.ann_dir, f"{cam}.txt"))
            for cam in self.video_paths
        }
        self.current_frame_ids = {cam: 0 for cam in self.video_paths}

        # Clear old canvases
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        screen_width = self.root.winfo_screenwidth()
        target_width = screen_width // 3
        target_height = int(target_width * 0.75)

        self.canvases = {}
        for idx, cam in enumerate(self.video_paths):
            self.video_sizes[cam] = (target_width, target_height)
            canvas = tk.Canvas(self.canvas_frame, width=target_width, height=target_height)
            row = idx // 3
            col = idx % 3
            canvas.grid(row=row, column=col, padx=5, pady=5)
            self.canvases[cam] = canvas
        answer = messagebox.askyesno("Resume?", "Resume from last saved frame positions?")
        if answer:
            self.load_last_frame()
        else:
            self.current_frame_ids = {cam: 0 for cam in self.frames}
            
        num_rows = 2
        # After placing all canvases (assume max 4 per row)
        label_frame = tk.Frame(self.canvas_frame)
        label_frame.grid(row=(num_rows * 2), column=0, columnspan=4, pady=5)

        for idx, cam in enumerate(self.video_paths):
            label = tk.Label(label_frame, text=f"{cam}: Frame 0 / {len(self.frames[cam])}", font=("Arial", 12))
            label.pack(side='left', padx=10)
            self.frame_labels[cam] = label

        # for idx, cam in enumerate(self.video_paths):
        #     self.video_sizes[cam] = (target_width, target_height)

        #     canvas = tk.Canvas(self.canvas_frame, width=target_width, height=target_height)
        #     row = idx // 3
        #     col = idx % 3
        #     canvas.grid(row=row * 2, column=col, padx=5, pady=5)  # Double row for label underneath
        #     self.canvases[cam] = canvas

        #     # Add label for frame info
        #     label = tk.Label(self.canvas_frame, text=f"{cam}: Frame 0 / {len(self.frames[cam])}")
        #     label.grid(row=row * 2 + 1, column=col)
        #     self.frame_labels[cam] = label
            
        self.show_all_frames()


    def show_all_frames(self):
        for cam in self.frames:
            frame_id = self.current_frame_ids[cam]
            canvas = self.canvases[cam]
            canvas.delete("all")  # Clear previous content

            if frame_id >= len(self.frames[cam]):
                canvas.create_text(100, 50, text="Frame not available", fill="red", font=("Arial", 14))
                continue

            frame_path = self.frames[cam][frame_id]
            frame = cv2.imread(frame_path)

            if frame is None:
                canvas.create_text(100, 50, text="Failed to load frame", fill="red", font=("Arial", 14))
                continue

            self.original_height, self.original_width = frame.shape[:2]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.video_sizes[cam])

            # Show image
            img = tk.PhotoImage(data=cv2.imencode('.png', frame)[1].tobytes())
            canvas.image = img
            canvas.create_image(0, 0, anchor=tk.NW, image=img)

            scale_x = self.video_sizes[cam][0] / self.original_width
            scale_y = self.video_sizes[cam][1] / self.original_height

            # Draw annotations
            if frame_id in self.annotations[cam]:
                for obj in self.annotations[cam][frame_id]:
                    x, y, w, h = obj['bbox']
                    x = int(x * scale_x)
                    y = int(y * scale_y)
                    w = int(w * scale_x)
                    h = int(h * scale_y)

                    canvas.create_rectangle(x, y, x + w, y + h, outline="chartreuse1", width=2)
                    canvas.create_text(x, y - 10, text=str(obj['id']), fill="firebrick1", font=("Arial", 19))
                    # gid = obj.get('global_id', obj['id'])
                    # canvas.create_text(x, y - 10, text=f"{obj['id']}|G{gid}", fill="firebrick2", font=("Arial", 16))
                    if frame_id % 10 == 0:
                        self.save_last_frame()
                    # annotated_count = len(self.annotations[cam])

            # Bind interaction events
            # At the end of canvas setup in show_all_frames():
            canvas.bind("<Button-3>", self.make_callback(self.on_click, cam))
            canvas.bind("<Button-1>", self.make_callback(self.start_draw, cam))
            canvas.bind("<B1-Motion>", self.make_callback(self.draw_rectangle, cam))
            canvas.bind("<ButtonRelease-1>", self.make_callback(self.finish_draw, cam))
            canvas.bind("<Button-2>", self.make_callback(self.select_box_for_move, cam))
            canvas.bind("<B2-Motion>", self.drag_box_move)
            canvas.bind("<ButtonRelease-2>", self.finish_drag_move)
            canvas.bind("</>", self.save_annotations)
            total = len(self.frames[cam])
            self.frame_labels[cam].config(text=f"{cam}: Frame {frame_id + 1} / {total}")
    
    def make_callback(self, func, cam):
        return lambda event: func(event, cam)

    
    def select_box_for_move(self, event, cam):
        frame_id = self.current_frame_ids[cam]
        if frame_id not in self.annotations[cam]:
            return

        # Scale to original resolution
        cap = self.video_caps[cam]
        scale_x = self.original_width / self.video_sizes[cam][0]
        scale_y = self.original_height / self.video_sizes[cam][1]

        x_click = event.x * scale_x
        y_click = event.y * scale_y

        closest_obj = None
        min_dist = float('inf')

        # Find closest bounding box to the click
        for obj in self.annotations[cam][frame_id]:
            x, y, w, h = obj['bbox']
            if x <= x_click <= x + w and y <= y_click <= y + h:
                dist = abs((x + w/2) - x_click) + abs((y + h/2) - y_click)
                if dist < min_dist:
                    closest_obj = obj
                    min_dist = dist

        if closest_obj:
            self.dragging_obj = closest_obj
            self.dragging_cam = cam
            self.drag_start = (event.x, event.y)

    def drag_box_move(self, event):
        if self.dragging_obj:
            dx = event.x - self.drag_start[0]
            dy = event.y - self.drag_start[1]
            self.drag_start = (event.x, event.y)

            cap = self.video_caps[self.dragging_cam]
            scale_x = self.original_width / self.video_sizes[self.dragging_cam][0]
            scale_y = self.original_height / self.video_sizes[self.dragging_cam][1]

            self.dragging_obj['bbox'][0] += int(dx * scale_x)
            self.dragging_obj['bbox'][1] += int(dy * scale_y)

            self.show_all_frames()

    def finish_drag_move(self, event):
        self.dragging_obj = None
        self.dragging_cam = None


    def remove_duplicate_annotations(self, camera, frame_id, iou_threshold=0.4):
        if frame_id not in self.annotations[camera]:
            return

        unique_objs = []
        for obj in self.annotations[camera][frame_id]:
            keep = True
            for u in unique_objs:
                if self.compute_iou(obj['bbox'], u['bbox']) > iou_threshold:
                    keep = False
                    break
            if keep:
                unique_objs.append(obj)

        self.annotations[camera][frame_id] = unique_objs

    def start_draw(self, event, cam):
        self.drawing = True
        self.start_x, self.start_y = event.x, event.y
        self.rectangles[cam] = self.canvases[cam].create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red")

    def draw_rectangle(self, event, cam):
        if self.drawing:
            canvas = self.canvases[cam]
            canvas.coords(self.rectangles[cam], self.start_x, self.start_y, event.x, event.y)

    def finish_draw(self, event, cam):
        if not self.drawing:
            return
        self.drawing = False

        end_x, end_y = event.x, event.y
        canvas = self.canvases[cam]
        canvas.delete(self.rectangles[cam])

        # Normalize coordinates
        x0, y0 = min(self.start_x, end_x), min(self.start_y, end_y)
        x1, y1 = max(self.start_x, end_x), max(self.start_y, end_y)
        w, h = x1 - x0, y1 - y0

        # Convert to original resolution
        orig_w = self.original_width
        orig_h = self.original_height
        scale_x = orig_w / self.video_sizes[cam][0]
        scale_y = orig_h / self.video_sizes[cam][1]
        real_x = int(x0 * scale_x)
        real_y = int(y0 * scale_y)
        real_w = int(w * scale_x)
        real_h = int(h * scale_y)

        obj_id = simpledialog.askinteger("New Object", "Enter ID for new object:")
        if obj_id is None:
            return

        frame_id = self.current_frame_ids[cam]
        if frame_id not in self.annotations[cam]:
            self.annotations[cam][frame_id] = []
        self.annotations[cam][frame_id].append({'id': obj_id, 'bbox': [real_x, real_y, real_w, real_h]})
        self.show_all_frames()


    def on_click(self, event, camera):
        frame_id = self.current_frame_ids[camera]
        x, y = event.x, event.y

        # Convert to original resolution
        orig_w = self.original_width
        orig_h = self.original_height
        scale_x = orig_w / self.video_sizes[camera][0]
        scale_y = orig_h / self.video_sizes[camera][1]

        real_x = int(x * scale_x)
        real_y = int(y * scale_y)

        if frame_id not in self.annotations[camera]:
            return

        updated = False
        new_objs = []

        for obj in self.annotations[camera][frame_id]:
            bx, by, bw, bh = obj['bbox']
            if bx <= real_x <= bx + bw and by <= real_y <= by + bh:
                # Found the box clicked
                choice = messagebox.askquestion(
                    "Modify Object",
                    f"ID: {obj['id']}\n Yes for (Eidt) ID and No for (delete) Bbox?",
                    icon='question',
                    type='yesnocancel',
                    default='yes'
                )
                if choice == 'yes':
                    new_id = simpledialog.askinteger("Edit ID", "Enter new ID:")
                    if new_id is not None:
                        obj['id'] = new_id
                        updated = True
                        new_objs.append(obj)
                elif choice == 'no':
                    updated = True  # Delete it by skipping append
                else:
                    new_objs.append(obj)  # Cancel: Keep unchanged
            else:
                new_objs.append(obj)

        if updated:
            self.annotations[camera][frame_id] = new_objs
            self.remove_duplicate_annotations(camera, frame_id)
            self.show_all_frames()






    def next_frame(self):
        for cam in self.current_frame_ids:
            self.current_frame_ids[cam] += 1
        self.show_all_frames()

    def prev_frame(self):
        for cam in self.current_frame_ids:
            if self.current_frame_ids[cam] > 0:
                self.current_frame_ids[cam] -= 1
        self.show_all_frames()
        
    def save_annotations(self):
        if not self.annotations:
            messagebox.showwarning("Warning", "No annotations to save.")
            return

        for cam in self.annotations:
            out_path = os.path.join(self.out_dir, f"{cam}.txt")
            save_mot_annotations(out_path, self.annotations[cam])
            print(f"Saved annotations for {cam} to {out_path}")

        messagebox.showinfo("Success", f"Annotations saved to {self.out_dir}")



# Annotation functions
def load_mot_annotations(path):
    annotations = {}
    with open(path, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split(',')))
            frame_id = int(parts[0])
            obj_id = int(parts[1])
            bbox = list(map(int, parts[2:6]))  # This should be a list of 4 ints
            if frame_id not in annotations:
                annotations[frame_id] = []
            annotations[frame_id].append({'id': obj_id, 'bbox': bbox})
    return annotations



def save_mot_annotations(path, annotations):
    with open(path, 'w') as f:
        for frame_id in sorted(annotations.keys()):
            for obj in annotations[frame_id]:
                x, y, w, h = obj['bbox']
                line = f"{frame_id},{obj['id']},{x},{y},{w},{h},1,-1,-1,-1\n"
                f.write(line)




# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1200x800")  # Start window size bigger for scrolling

    app = MOTAnnotationEditor(root)
    root.mainloop()
