It is better to use p2 quantisation in this model, it gives better accuracy!
With power of two quantization, we get an accuracy of 47.95 if flip test isn't used, whereas 52.16 if flip test is used. 
With non power of two quantization, we get an accuracy of 44.16 if flip test isn't used, whereas 48.63 if flip test is used. 
Further, the above accuracy are when first and last layers of the models are quantised to 16 bits and the rest of the layers in 8 bit. The first and last layer of the model are 675 and 1416.
This model was finetuned for 30 epochs on padded images with udp and the best accuracy was obtained on epoch 25.