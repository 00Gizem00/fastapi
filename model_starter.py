from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image


processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

def model_pipeline(text: str, image: Image):
    # prepare inputs
    encoding = processor(image, text, return_tensors="pt")

    # forward pass
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()

    return model.config.id2label[idx]



# this part is just printing the result to the console
# from transformers import ViltProcessor,  ViltForQuestionAnswering
# import requests
# from PIL import Image

# # prepare image + question
# url = "https://unsplash.com/photos/gray-and-black-mallard-ducks-flying-during-day-time-WPmPsdX2ySw"
# image = Image.open(requests.get(url, stream=True).raw)
# text = "What's the animal doing?"

# processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
# model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# # prepare inputs
# encoding = processor(image, text, return_tensors="pt")

# # forward pass
# outputs = model(**encoding)
# logits = outputs.logits
# idx = logits.argmax(-1).item()
# print("Predicted answer:", model.config.id2label[idx])
