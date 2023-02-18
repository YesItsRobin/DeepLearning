# Assignment 1 - Implement a simple Neural Network

RUN main.py or mainp2.py to start the software
----------------------------------------------
The difference of these files is wether you want to initialize the network with a 4 dimensional input and no secret layers or a 1 dimensional input with 1 secret layer. The output layer is always 1 dimensional and the other parameters like epochs and learning rate is the same. Because of the change in dimensions, the training data also looks different. If a different set of data is required, change the dataset in the corresponding file. 

The structure of the data is as follows: dataset = [ [input data1,expected output1], [input data2,expected output2], etc.. ]

The main file will initialize an instance of MyNetwork and call the train function to start training.