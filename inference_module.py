import cv2
import time

def inference_loop(model, frame_queue, stop_event, latest_detections,
                   class_names, frame_width, img_size, conf_thresh, iou_thresh):
    while not stop_event.is_set():
        if frame_queue.empty():
            time.sleep(0.01)
            continue

        frame = frame_queue.get()

        # Run YOLO inference
        results = model.predict(
            frame, imgsz=img_size, conf=conf_thresh, iou=iou_thresh, verbose=False
        )

        detections = []
        annotated_frame = frame.copy()

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                label = class_names[cls_id] if cls_id < len(class_names) else "unknown"
                conf = float(box.conf[0].item())
                track_id = int(box.id[0].item()) if box.id is not None else -1

                # Distance estimation (simple proportional formula)
                w = box.xyxy[0][2] - box.xyxy[0][0]
                dist = (0.4 * 600) / w if w > 0 else None

                detections.append((label, dist, track_id, conf))

                # --- Draw bounding boxes ---
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (0, 255, 0)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                text = f"{label} {conf:.2f}"
                if track_id != -1:
                    text = f"ID {track_id}: {text}"
                cv2.putText(annotated_frame, text, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Update shared detections
        latest_detections.clear()
        latest_detections.extend(detections)

        # Show live preview
        cv2.imshow("Smart Glasses - Live Preview", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    cv2.destroyAllWindows()
