class Analytica(object):
    WHO_LIST = []    #. classe Super nao fica registrada na WHOLIST.
    evaluations_all = {}
    
    def __init__(self, pipe):
        '''na declaracao da classe, o obj fica relacionado ao pipe podendo ser usado por qq funcao '''
        self.pipe = pipe    
#         self.sufix = sufix
# #         Analytica.insert_member(self)
#         Analytica.make_data(self)
        
        

    def insert_member(self):   # only used in the Sub init
#         Super_Analytica.WHO_LIST.append(self.my_name)  ## APAGAR DEPOIS
        if not self.my_name in Analytica.WHO_LIST:
            Analytica.WHO_LIST.append(self.my_name)
            print(self.my_name, '  Instance Created')
            print('\n')
        else:
            raise ValueError(self.my_name, ' This instance is already in WHOLIST. Or pick another name.')
            
    def make_data(self, sufix, pkl_W=False):
#         self.pipe = pipe 
        print('Vectorizing Data')
        self.X_train_ = self.pipe.fit_transform(X_train, y_train)  
        self.X_test_ = self.pipe.transform(X_test)
        self.X_val_ = self.pipe.transform(X_val)
        
        print(self.X_train_.shape)
        print(self.X_test_.shape)
        print(self.X_val_.shape)
        
        if pkl_W:
            filenames = ['X_train_', 'X_test_', 'X_val_']
            for name, data in zip(filenames, [self.X_train_, self.X_test_, self.X_val_]):
                joblib.dump(data, '../pkl/'+ name + sufix + '.pkl') 
            

        print('Pickle Done')
        
    
    def read_data(self, sufix):
        self.X_train_ = joblib.load('../pkl/'+ 'X_train_' + sufix + '.pkl') 
        self.X_test_ = joblib.load('../pkl/'+ 'X_test_' + sufix + '.pkl') 
        self.X_val_ = joblib.load('../pkl/'+ 'X_val_' + sufix + '.pkl') 
        print('Done Loading Data: ', 'X_train_' + sufix, '  X_test_' + sufix, '  X_val_' + sufix )   
    
   
    def get_members_evaluations_all(self):
        return Analytica.evaluations_all.keys() # pega somente os modelos que acionaram evaluate_HP
    
    
    def extract_model_instance(self, my_name):
        '''my_name must be a single value string. 
        Returns the model instance out of the pipeline. ( pipelilne stored in evaluaions all) 
        Intended to export the fitted model and it`s best params for reuse (bagging).
        '''
#         my_name_no_under = my_name.split('_')[0]
#         return Super_Analytica.evaluations_all[my_name]['best_estimator'].named_steps[my_name_no_under]
        return Analytica.evaluations_all[my_name]['best_estimator']


#######################.         PLOT            ####################       
#######################.         PLOT            ####################   
      
     
#     def Evaluate_HP_val(self, classifier, space, my_name, pkl_W=False):     
    def plot_learning_curve_2(self, load_saved=True, chosen=None):  
        '''
        If load_saved == False it will overwrite any file named equal
        chosen must be a list of members of evaluations_all
        if chosen == None, it will process all the models in evaluations all
        '''
        list_members = self.get_members_evaluations_all()
        
        if chosen:
            assert set(chosen).issubset(set(list_members)) # all chosen in evaluations_all
            list_members = chosen # check if chosen are in evaluations all
            list_members.sort()     
        
        for model in list_members :
            path = '../pkl/l_curve_'+ model + '.png'
            if load_saved == True:
#                 print(type(name))
                print('opening the image',  model)
        
                display(Image(path))
            else:
                
                mdl = Analytica.evaluations_all[model]['best_estimator']
#                 print(mdl)
                sizes = np.linspace(0.2, 1.0, 5)
                viz = LearningCurve(mdl, train_sizes=sizes, scoring='roc_auc')
                viz.fit(self.X_train_, y_train)
                viz.poof(outpath=path)
                print('saving l_curve_ image for ',  model)
                plt.clf()


    
#######################.         EVALUATE            ####################       
#######################.         EVALUATE            ####################                    

        # new
    def Evaluate_HP_val(self, classifier, space, my_name, pkl_W=False): 
        '''
            classifier must be a string representing a clf_all member. space must be a string representing a key from space_all.
            Essa funcao realiza parameter tunning via validation set e aplica best_model em todo dataset de treino.
        '''
        self.my_name = my_name
        Analytica.insert_member(self)
        start= timer()

        self.instance = clone(clf[classifier])  #  to avoid edit the original object
#         print('self.instance0: ', self.instance)
           
        def objective(params):
#             print('self.instance1: ', self.instance, '\n')
            self.instance.set_params(**params)  
#             print('self.instance2: ', self.instance, '\n')
            mdl = self.instance.fit(self.X_train_, y_train) # n_jobs=-1) 
            y_pred_val = mdl.predict(self.X_val_)
            score = roc_auc_score(y_val, y_pred_val)
            print('val score: ', score)
            return 1 - score
        
            
        T = Trials()
        space_ = space_all[space] 
        best = fmin(objective, space_, algo=tpe.suggest, max_evals=200, trials=T)  #  <---------------------------HERE<-----
        best_params = space_eval(space_, best) # is the translation of best. Real Values, not steps.
#         print('best_params :', best_params, '\n')
        best_model = self.instance.set_params(**best_params)
        best_model.fit(self.X_train_, y_train) 
#         print('best_model :', best_model, '\n')
        
        y_pred = best_model.predict_proba(self.X_test_)[:, 1]
        acc = roc_auc_score(y_test, y_pred) 
        print('acc :', acc, '\n')
        
        pipe_best = clone(self.pipe)
        pipe_best.steps.append([self.my_name, best_model]) # params so do modelo
#         print('pipe_best: ', pipe_best)
        
        end = timer()
        minutes_elapsed = (end - start)//60
        Analytica.evaluations_all[self.my_name] = {'score': acc, 'best_estimator': best_model, \
                                 'param_grid': space, 'y_pred': y_pred, \
                                 'pipe_best': pipe_best, 'instance':self.instance,\
                                 'best_params' : best_params, 'T': T, 'minutes': minutes_elapsed} 
                

        if  pkl_W:
            D = {}
            D[self.my_name]= Analytica.evaluations_all[self.my_name]
            joblib.dump(D, '../pkl/clf_'+ self.my_name + '.pkl') 
            print(self.my_name, ' HP_val and pickled to disk Done in',  minutes_elapsed, ' minutes' ) 
        else:
            print(self.my_name, '  HP_val done in ', minutes_elapsed, ' minutes') 



        # new
    def Evaluate_HP_cv(self, classifier, space,  my_name, pkl_W=False): 
      
        self.my_name = my_name
        Analytica.insert_member(self)
        start= timer()
        result_dict = {}

        self.instance = clone(clf[classifier])  #  to avoid edit the original object
#         print('self.instance: ', self.instance)

        def objective(params):
#             pipe_full.set_params(**params)  # diferente do caso sem pipe
            self.instance.set_params(**params)  
#             print('self.instance: ', self.instance)
            score = cross_val_score(self.instance, self.X_train_, y_train, scoring='roc_auc', cv=kfold) # n_jobs=-1) 
#             print('cv score:', score.mean())
            return 1 - score.mean()


            
        T = Trials()
        best = fmin(objective, space, algo=tpe.suggest, max_evals=40, trials=T)  #  <---------------------------HERE<-----
        best_params = space_eval(space, best) # is the translation of best. Real Values, not steps.
#         print('best_params :', best_params, '\n')
        best_model = self.instance.set_params(**best_params)
        best_model.fit(self.X_train_, y_train) 
#         print('best_model :', best_model, '\n')
        
        y_pred = best_model.predict_proba(self.X_test_)[:, 1]
        acc = roc_auc_score(y_test, y_pred) 
        print('acc :', acc, '\n')
        
        pipe_best = clone(self.pipe)
#         print('self.pipe: ',  self.pipe)
        pipe_best.steps.append([self.my_name, best_model]) # params so do modelo
#         print('pipe_best: ', pipe_best)
        
        end = timer()
        minutes_elapsed = (end - start)//60
        Analytica.evaluations_all[self.my_name] = {'score': acc, 'best_estimator': best_model, \
                                 'param_grid': None, 'y_pred': y_pred, \
                                 'pipe_best': pipe_best, 'instance':self.instance,\
                                 'best_params' : best_params, 'T': T, 'minutes': minutes_elapsed} 
                

        if  pkl_W:
            D = {}
            D[self.my_name]= Analytica.evaluations_all[self.my_name]
            joblib.dump(D, '../pkl/clf_'+ self.my_name + '.pkl') 
            print(self.my_name, ' HP_cv and pickled to disk Done in',  minutes_elapsed, ' minutes' ) 
        else:
            print(self.my_name, '  HP_cv done in ', minutes_elapsed, ' minutes') 

 