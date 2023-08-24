import KnowledgeGraphPI as kg
import cv2
import torch
import numpy as np
import networkx as nx


def main(image_path: str, input_graphML_path: str=None):
    flowsheet_graphml = kg.from_grapml(input_graphML_path)
    flowsheet_image   = cv2.imread(image_path)[:-1]
    model = torch.load('model_final.pth', map_location=torch.device('cpu'))
    model.eval()
    torch.inference_mode()
    # get the instances from the graphml
    for obj_ in flowsheet_graphml.objects:
        cropped_instance = crop_instance(obj_, flowsheet_image)
        # run the model on the instance depiction
        model_output = run_model(cropped_instance, model)
        flip = False
        if model_output > 3:
            flip = True
        if model_output == 0 or model_output == 4:
            rotation = 0
        elif model_output == 1 or model_output == 5:
            rotation = 90 
        elif model_output == 2 or model_output == 6:
            rotation = 180
        else:
            rotation = 270
        obj_orientation = kg.Orientation(k=0, flip=flip, rotation=rotation)
        obj_.orientation[0] = obj_orientation
        
        
    flowsheet_graph = flowsheet_graphml.to_networkX()
    kg.to_graphml(flowsheet_graph, 'output_graphml.graphml')    
    g_string = ""
    for line in nx.generate_graphml(flowsheet_graph, prettyprint=True, named_key_ids=True):
        g_string += line
    return g_string
    
def crop_instance(obj, flowsheet_image):
    instance_image = flowsheet_image[int(obj.bounding_box.y_min):int(obj.bounding_box.y_max),int(obj.bounding_box.x_min):int(obj.bounding_box.x_max)]                     
    return instance_image
    
def run_model(instance_image, model):
    # preprocess the instance image (resize)
    image = cv2.resize(instance_image, (224, 224))
    image = image.astype("float32") / 255.0
    image = np.transpose(image, (2, 0,  1))
    image = torch.from_numpy(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        preds = model(image)
        preds = torch.argmax(preds, dim=1)
    return preds
    
if __name__ == '__main__':
    main()

    