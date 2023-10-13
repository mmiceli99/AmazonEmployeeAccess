library(vroom)
library(tidyverse)
library(tidymodels)
library(embed)
library(ggmosaic)

ama_train <- vroom("C:/School/Stat348/AmazonEmployeeAccess/train.csv")
ama_test <- vroom("C:/School/Stat348/AmazonEmployeeAccess/test.csv")

ggplot(ama_train) +
  geom_bar(aes(x=ama_train$ACTION))

ggplot(ama_train) +
  geom_boxplot(aes(x=ama_train$RESOURCE, y = ACTION))

ggplot(data=ama_train) +
  geom_mosaic(aes(x=ama_train$ROLE_TITLE), fill=ama_train$ACTION)
#mmiceli99