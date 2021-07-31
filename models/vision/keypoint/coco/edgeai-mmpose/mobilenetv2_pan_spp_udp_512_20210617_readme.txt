It is better to use np2 quantisation in this model, it gives better accuracy!
With power of two quantization, we get an accuracy of 30.27 if flip test isn't used, whereas 33.91 if flip test is used. 
With non power of two quantization, we get an accuracy of 39.84 if flip test isn't used, whereas 44.27 if flip test is used. 
Further, the above accuracy are when first and last layers of the models are quantised to 16 bits and the rest of the layers in 8 bit. The first and last layer of the model are 669 and 1384.
This model was finetuned for 38 epochs on padded images with udp and the best accuracy was obtained on epoch 32.