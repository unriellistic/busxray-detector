from detectron2.structures import Instances

def parse_inference(inference: Instances) -> list[dict]:
    '''Convert the raw detectron2 inference format into a more readable format.'''
    parsed = []
    fields = inference.get_fields()
    for i in range(len(inference)):
        parsed.append({
            "bbox": fields["pred_boxes"].tensor.cpu().numpy().tolist()[i],
            "score": fields["scores"].cpu().numpy().tolist()[i],
            "pred_class": fields["pred_classes"].cpu().numpy().tolist()[i]
        })
    
    return parsed