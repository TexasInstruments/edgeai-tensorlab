It is better to use np2 quantisation in this model, it gives better accuracy!
With power of two quantization, we get an accuracy of 30.65 if flip test isn't used, whereas 35.50 if flip test is used. 
With non power of two quantization, we get an accuracy of 38.72 if flip test isn't used, whereas 43.28 if flip test is used. 
Further, the above accuracy are when first and last layers of the models are quantised to 16 bits and the rest of the layers in 8 bit. The first and last layer of the model are 363 and 561.
This model was finetuned for 20 epochs on padded images with udp and the best accuracy was obtained on epoch 16.