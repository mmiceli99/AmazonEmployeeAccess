library(vroom)
library(tidyverse)
library(tidymodels)
library(embed)

ama_train <- vroom("C:/School/Stat348/AmazonEmployeeAccess/train.csv")
ama_test <- vroom("C:/School/Stat348/AmazonEmployeeAccess/test.csv")

# ggplot(ama_train) +
#   geom_bar(aes(x=ama_train$ACTION))
# 
# ggplot(ama_train) +
#   geom_box(aes(x=ama_train$RESOURCE, y = ACTION))
# 
ama_train <- ama_train %>%
  mutate(ACTION = as.factor(ACTION))
my_recipe <- recipe(ACTION ~ ., data=ama_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  #step_mutate_at(ACTION, fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .01) %>% # combines categorical values that occur <5% into an "other" value
  step_dummy(all_nominal_predictors()) %>% # dummy variable encoding
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) #target encoding
# also step_lencode_glm() and step_lencode_bayes()


# NOTE: some of these step functions are not appropriate to use together

# apply the recipe to your data
prep <- prep(my_recipe)
baked <- bake(prep, new_data = ama_train)



# Fit logistic regression model
my_mod <- logistic_reg() %>%
  set_engine('glm')


ama_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod) %>%
  fit(data=ama_train)

ama_predictions <- predict(ama_workflow, new_data=ama_test, type='prob') %>%
  mutate( Id = row_number()) %>%
  rename(Action=.pred_1) %>%
  select(Id, Action)
vroom_write(x=ama_predictions, file="./LogReg.csv", delim=",")

