It is better to use np2 quantisation in this model, it gives better accuracy!
With power of two quantization, we get an accuracy of 42.58 if flip test isn't used, whereas 47.13 if flip test is used. 
With non power of two quantization, we get an accuracy of 44.79 if flip test isn't used, whereas 50.58 if flip test is used. 
Further, the above accuracy are when first and last layers of the models are quantised to 16 bits and the rest of the layers in 8 bit. The first and last layer of the model are 369 and 590.
This model was finetuned for 20 epochs on padded images with udp and the best accuracy was obtained on epoch 19.