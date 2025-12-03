person_boxes = []
vest_boxes = []
helmet_boxes = []
boots_boxes = []
gloves_boxes = []

def define_boxes(boxes, model):
    global person_boxes, vest_boxes, helmet_boxes, boots_boxes, gloves_boxes

    class_names = model.names if hasattr(model, "names") else {}
    person_boxes.clear()
    vest_boxes.clear()
    helmet_boxes.clear()
    boots_boxes.clear()
    gloves_boxes.clear()

    for box in boxes:
        class_id = int(box[5])
        label = class_names.get(class_id, str(class_id)).lower()
        if label == "person":
            person_boxes.append(box)
        elif label == "vest":
            vest_boxes.append(box)
        elif label == "helmet":
            helmet_boxes.append(box)
        elif label == "boots":
            boots_boxes.append(box)
        elif label == "gloves":
            gloves_boxes.append(box)

    return person_boxes, vest_boxes, helmet_boxes, boots_boxes, gloves_boxes
