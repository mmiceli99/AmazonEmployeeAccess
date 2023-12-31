library(vroom)
library(tidyverse)
library(tidymodels)
library(embed)
library(discrim)
library(naivebayes)
library(doParallel)

#parallel::detectCores() #How many cores do I have?
#cl <- makePSOCKcluster(4) # num_cores to use
#registerDoParallel(cl)


ama_train <- vroom("./train.csv")
ama_test <- vroom("./test.csv")

ama_train <- ama_train %>%
  mutate(ACTION = as.factor(ACTION))
my_recipe <- recipe(ACTION ~ ., data=ama_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  #step_mutate_at(ACTION, fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <5% into an "other" value
  #step_dummy(all_nominal_predictors()) %>% # dummy variable encoding
  
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), threshold=.9) #Threshold is between 0 and 1#target encoding
# also step_lencode_glm() and step_lencode_bayes()


# NOTE: some of these step functions are not appropriate to use together

# apply the recipe to your data
prep <- prep(my_recipe)
baked <- bake(prep, new_data = ama_train)

nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") # install discrim library for the naivebayes eng

nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)

## Grid of values to tune over
tuning_grid <- grid_regular(Laplace(),
                            smoothness(),
                            levels = 5) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(ama_train, v = 5, repeats=1)

## Run the CV
CV_results <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("roc_auc")

## Finalize the Workflow & fit it
final_wf <-
  nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=ama_train)

## Predict
ama_predictions <- predict(final_wf, new_data=ama_test, type='prob') %>%
  mutate( Id = row_number()) %>%
  rename(Action=.pred_1) %>%
  select(Id, Action)
vroom_write(x=ama_predictions, file="./PCAnaiveBayes.csv", delim=",")
#stopCluster(cl)
