class Analytica(object):
    
    def __init__(self, instance, param_grid, my_name):
        self.my_name = my_name
        self.instance = instance    
        self.param_grid = param_grid
        Super_Analytica.insert_member(self)
        
 
 
    def evaluate_HP_cv(self, pipe, pkl_W=False): 
        '''       
            Essa funcao realiza parameter tunning via cross validation e aplica best_model em todo dataset de treino.
            O parametro 'pipe' nao deve conter um classificador como ultimo elemento.
        '''
        start= timer()
        result_dict = {}

        instance = self.instance
        param_grid = self.param_grid

        pipe_full = clone(pipe)
        pipe_full.steps.append([instance, clf[instance]])

       
        def objective(params):
            pipe_full.set_params(**params) 
            score = cross_val_score(pipe_full, X_train, y_train, scoring='roc_auc', cv=kfold)
            print('cv score:', score.mean())
            return 1 - score.mean()
        
        T = Trials()
        space = param_grid
        best = fmin(objective, space, algo=tpe.suggest, max_evals=30, trials=T)  #  <---------------------------HERE<-----
        best_params = space_eval(space, best) # is the translation of best. Real Values, not steps.
        pipe_full.set_params(**best_params)
        clf_ = pipe_full.fit(X_train, y_train) 

        y_pred = clf_.predict_proba(X_test)[:, 1]
        acc = roc_auc_score(y_test, y_pred) 
        print('test score :', acc)
        
        end = timer()
        minutes_elapsed = (end - start)//60
        
        # record all scores to evaluations_all
        Super_Analytica.evaluations_all[self.my_name] = {'score': acc, 'best_estimator': clf_, \
                                 'param_grid': param_grid, 'y_pred': y_pred, \
                                 'pipe_best': pipe_full, 'instance':instance,\
                                 'best_params' : best_params, 'T': T, 'minutes': minutes_elapsed} 
                

        if  pkl_W:
            D = {}
            D[self.my_name]= Super_Analytica.evaluations_all[self.my_name]
            joblib.dump(D, '../pkl/clf_'+ self.my_name + '.pkl') 
            print(self.my_name, ' HP_cv and pickled to disk Done in',  minutes_elapsed, ' minutes' ) 
        else:
            print(self.my_name, '  HP_cv done in ', minutes_elapsed, ' minutes') 
      
    

            
######################        PARFIT       ######################
            
    def evaluate_Parfit(self, pipe, pkl_W=False): 
        '''
        Essa funcao realiza parameter tunning via validation set e aplica best_model em todo dataset treino.
        O pipeline 'pipe' nao deve conter um classificador como ultimo elemento.
        X_train, X_val, X_test serao vetorizados de acordo com o parametro 'pipe'.
        O praram_grid deve conter ParameterGrid({})
            
        '''
        start= timer()
        instance = self.instance
        param_grid = self.param_grid        

        X_train_ = pipe.fit_transform(X_train, y_train) 
        X_val_ = pipe.transform(X_val)
        X_test_ = pipe.transform(X_test)
        
        best_model, best_score, all_models, all_scores = pf.bestFit(clf[instance], \
                                                                     
            param_grid, X_train_, y_train, X_val_, y_val,\
            metric=roc_auc_score, scoreLabel='AUC')
#         print(best_model)
        
        mdl = best_model.fit(X_train_, y_train) 
        y_pred = mdl.predict_proba(X_test_)[:, 1]
        acc = roc_auc_score(y_test, y_pred) 
        print('acc :', acc)

        
        pipe_best = clone(pipe)
        pipe_best.steps.append([self.my_name, best_model])

        end = timer()
        minutes_elapsed = (end - start)//60
        Super_Analytica.evaluations_all[self.my_name] = {'score': acc, 'best_estimator': best_model , \
                                 'param_grid': param_grid, 'y_pred': y_pred, \
                                 'pipe_best': pipe_best, 'instance':instance,\
                                 'best_params' : best_model, 'T': None, 'minutes': minutes_elapsed} 
          
        print('start pickling')
        if pkl_W:
            D = {}
            D[self.my_name]= Super_Analytica.evaluations_all[self.my_name]
            joblib.dump(D, '../pkl/clf_'+ self.my_name + '.pkl')     
            print(self.my_name, ' Parfit and pickled to disk Done in',  minutes_elapsed, ' minutes' ) 
        else:
            print(self.my_name, '  Parfit done in ', minutes_elapsed, ' minutes') 
        
    
    
######################        LIGHT GBM         ########################
         
    
    def evaluate_HP_lGBM(self, pipe, pkl_W=False): 
        '''
            This function uses HyperOpt to tune LightGHM params on train and validation set. Than it refits the best model on entire trainset and score on testset.
            O pipeline 'pipe' nao deve conter um classificador como ultimo elemento.
        '''
        start= timer()
        result_dict = {}

        instance = self.instance
        param_grid = self.param_grid


        def objective(params):
            # Make sure parameters that need to be integers are integers
            for parameter_name in ['num_leaves', 'subsample_for_bin', 'min_child_samples']:
                params[parameter_name] = int(params[parameter_name])

            model = clf[instance].set_params(**params)
            model.fit(train_feature_matrix, y_train, eval_set=(val_feature_matrix, y_val),\
                      eval_names=['val_feature_matrix', 'y_val'], eval_metric='auc', early_stopping_rounds=10, verbose=False)             
            best_score = model.best_score_['val_feature_matrix']['auc']
#             print('val score: ', best_score)
            return 1 - best_score


        train_feature_matrix = pipe.fit_transform(X_train, y_train) # pipe descalco sem o ultimo
        data_train = lgb.Dataset(train_feature_matrix, label=list(y_train.values.ravel()))

        test_feature_matrix = pipe.transform(X_test) # pipe sem o ultimo
        data_test = lgb.Dataset(test_feature_matrix, label=list(y_test.values.ravel()))
        
        val_feature_matrix = pipe.transform(X_val) 

        T = Trials()
        space = param_grid
        best = fmin(objective, space, algo=tpe.suggest, max_evals=30, trials=T)  #  <---------------------------HERE<-----
        best_params = space_eval(space, best) 

        for parameter_name in ['num_leaves', 'subsample_for_bin', 'min_child_samples']:
                best_params[parameter_name] = int(best_params[parameter_name])

        best_model = clf[instance].set_params(**best_params)
        best_model.fit(train_feature_matrix, y_train.values.ravel())
        y_pred = best_model.predict_proba(test_feature_matrix)[:,1]
#         y_pred = best_model.predict(test_feature_matrix)
#         print(y_pred)
        acc = roc_auc_score(y_test, y_pred) 
        print('test score :', acc)
        
        pipe_best = clone(pipe)
        pipe_best.steps.append([self.my_name, best_model])
#         print('pipe_best :', pipe_best.steps)
        
        end = timer()
        minutes_elapsed = (end - start)//60
        Super_Analytica.evaluations_all[self.my_name] = {'score': acc, 'best_estimator': best_model, \
                                 'param_grid': param_grid, 'y_pred': y_pred, 
                                 'pipe_best': pipe_best, 'instance':instance, # é somente uma string, nao é a instance em si\ 
                                 'best_params' : best_params, 'T': T, 'minutes': minutes_elapsed} 


        if  pkl_W:
            D = {}
            D[self.my_name]= Super_Analytica.evaluations_all[self.my_name]
            joblib.dump(D, '../pkl/clf_'+ self.my_name + '.pkl') 
            print(self.my_name, ' HP_lightGBM and pickled to disk Done in',  minutes_elapsed, ' minutes' ) 
        else:
            print(self.my_name, '  HP_lightGBM done in ', minutes_elapsed, ' minutes') 



