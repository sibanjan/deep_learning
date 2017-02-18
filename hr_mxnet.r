library(mxnet)
hr_data <- read.csv("F:/git/deep_learning/mxnet/hrdata/HR.csv")
head(hr_data)
str(hr_data)
summary(hr_data)
#Convert some variables to factors
hr_data$sales<-as.factor(hr_data$sales)
hr_data$salary<-as.factor(hr_data$salary)
hr_data$Work_accident <-as.factor(hr_data$Work_accident)
hr_data$promotion_last_5years <-as.factor(hr_data$promotion_last_5years)
smp_size <- floor(0.70 * nrow(hr_data))
## set the seed to make your partition reproductible
set.seed(1)
train_ind <- sample(seq_len(nrow(hr_data)), size = smp_size)

train <- hr_data[train_ind, ]
test <- hr_data[-train_ind, ]

train.preds <- data.matrix(train[,! colnames(hr_data) %in% c("left")])
train.target <- train$left
head(train.preds)
head(train.target)

test.preds <- data.matrix(test[,! colnames(hr_data) %in% c("left")])
test.target <- test$left
head(test.preds)
head(test.target)

#set seed to reproduce results
mx.set.seed(1)

mlpmodel <- mx.mlp(data = train.preds
                    ,label = train.target
                    ,hidden_node = c(3,2) #two layers- 1st Layer with 3 nodes and 2nd with 2 nodes
                    ,out_node = 2 #Number of output nodes
                    ,activation="sigmoid" #activation function for hidden layers
                    ,out_activation = "softmax" 
                    ,num.round = 10 #number of iterations
                    ,array.batch.size = 5 #Batch size for updating weights
                    ,learning.rate = 0.03 #same as step size
                    ,eval.metric= mx.metric.accuracy
                    ,eval.data = list(data = test.preds, label = test.target))
					
 #make a prediction
preds <- predict(mlpmodel, test.x)
dim(preds)

#configure the network structure
data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name = "fc1", num_hidden=3) #1st hidden layer with activation function sigmoid
act1 <- mx.symbol.Activation(fc1, name = "sig", act_type="sigmoid") 
fc2 <- mx.symbol.FullyConnected(act1, name = "fc2", num_hidden=2) #2nd hidden layer with activation function relu
act2 <- mx.symbol.Activation(fc2, name = "relu", act_type="relu") 
out <- mx.symbol.SoftmaxOutput(act2, name = "soft")

#train the network
dp_model <- mx.model.FeedForward.create(symbol = out
                                         ,X = train.preds
                                         ,y = train.target
                                         ,ctx = mx.cpu()
                                         ,num.round = 10
                                         ,eval.metric = mx.metric.accuracy
                                         ,array.batch.size = 50
                                         ,learning.rate = 0.005
 ,eval.data = list(data = test.preds, label = test.target))
 
 graph.viz(mlpmodel$symbol$as.json())
 
 #make a prediction
preds <- predict(dp_model, test.preds)
preds.target <- apply(data.frame(preds[1,]), 1, function(x) {ifelse(x >0.5, 1, 0)})
table(test.target,preds.target)

 