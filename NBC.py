import pandas as pd #inorder to read csv file
import numpy as np
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

class NBC(object):

  def __init__(self, f_num=5, cropus_root="/content/drive/MyDrive/Colab Notebooks/School related/Naive Bayesian Classifier/data/Glass/", file_="glass.data"):
    '''
    
    YOU SHOULD CHANGE THE ROOT WITH YOUR OWN PATH
    
    import a particular data set at one class, so ww'll have to run at least 4 times 
    (to generate new classes for different dataset)
    all of the data come from google drive

    Only process the missing value when dataset is hepatits
    once an instance have missing value, ignore the particular value and calculate the others value's probability
    !!!!!!! class will located at -1 !!!!!!!! <--- I change the Class index into -1, but the 'Class' column name must be 'Class', so the program can run it in right way
    !!!!!!! NOT CONSIDER MISSING VALUE !!!!!!!!
    '''
    self.path = cropus_root + file_
    self.df_before = pd.read_csv(self.path)
    self.f_num = f_num # init with 5
    
    #####ATTRIBUTES & CLASS PREPARE below
    self.attributes_count = 0 # including class
    self.attribute_names = [] # bcuz we'll transfer the attributes name into int
    self.attributes_code_pairs = {} # "age":0, "sex":1, "class":2...

    ##### change the class index into the last index
    temp_AttSe = [] # use it to store attributes sequence
    for i in self.df_before:
      temp_AttSe.append(i) # input the attributes and class name
    tempclass = temp_AttSe.index('Class') # find the class index
    if tempclass == len(temp_AttSe)-1: # class is at the last index
      list_without_class = temp_AttSe[0:tempclass]
    else:
      list_without_class = temp_AttSe[tempclass+1:len(temp_AttSe)] 
    list_without_class.append('Class') # add class at the end of this array
    self.df = self.df_before.reindex(columns=list_without_class) # reindex this df
    
    ##### count the numbers of the attributes and class in this df
    for i in self.df: 
      self.attributes_count += 1 # i.e. is 3
      self.attribute_names.append(i) # i.e. ['age','sex','Class']

    ##### change the attributes names into 0 to end
    self.df.columns = range(0, self.attributes_count) 
    for i,v in enumerate(self.attribute_names): # "age":0, "sex":1, "class":2...
      self.attributes_code_pairs[v] = i 
    
  def main(self):
    '''
    This is the main function of this classifier, including fit() and test().
    This function will return(or print) the accuracy of train state and test state.
    '''
    self.discretization() # return the value after discretization, process all of the continuous calumn
    print(self.df.head())
    self.foldlization() # split dataset into folds
    ans_attrs, ans_acc = self.SNB() # do the SNB test and return the final accuracy and the final attrs
    
    return ans_attrs, ans_acc # attrs and accuracy

  def discretization(self):
    '''
    process:

    for every attributes:
    try:
      do discretization with no missing value
    except ValueError:
      do discretization with missing value
    '''
    for attr in self.df:
      temp = self.df[attr].tolist()
      before = temp # use it to map, don't do anything on 'before' list!
      if attr != self.attributes_count-1: # avoid class column
        temp = list(map(float, temp))
        if len(set(temp)) > 10 : # do discretization
          step = (max(temp) - min(temp))/10 # to calculate bins step
          bins = []
          for i in range(0, 11):
            if i == 0:
              bins.append(min(temp)-0.0001) # def bins 
            elif i < 10:
              bins.append(min(temp)+step*i)
            else:
              bins.append(max(temp))
          print('bins',bins)
          label = [i for i in range(0,10)] # transfer into 0 to 9
          after = pd.cut(temp, bins, labels=label) # after discretization
          pair = {}
          for index, value in enumerate(before):
            pair[value] = after[index]
          self.df[attr] = self.df[attr].map(pair) # replace the old value

        else: # no need to do discretization, but still replace with the float value
          pair = {}
          for index, value in enumerate(before):
            pair[value] = temp[index]
          self.df[attr] = self.df[attr].map(pair) # use float to replace the old value
    
  def foldlization(self):
    # set the self.k_index as index of data []
    self.df = self.df.sample(frac=1) # mix the instances
    length = len(self.df)
    step = int(length/self.f_num)
    remain = length - step * self.f_num # it's 5 currently
    truee = []
    for i in range(1, self.f_num+1):
      truee.append(i * step)
    for i in range(0, remain):
      truee[(self.f_num-1) - i] += (remain-i)
    self.k_index = truee

  def SNB(self):
    '''
    process:
    1.use final list to store gerentee attrs and acc
    2.use acc_list to store the test acc, and pick the max one as a member of final list
    '''
    final_attr_list = [] # to store the attributes input aequence. i.e. [4, 11, 0]
    final_acc_list = [] # to store the accuracy of each loop in best case i.e. [0.7, 0.78, 0.89]

    for loop in range(0, self.attributes_count-1): # not containing class
      acc_list = [] # to store the accuracy of every attr add in set
      print('loop-------', loop)
      # so if we have 5 attrs remaining(not in the set), len(acc_list) will be 5
      for attrs in self.df: # containing class
        if attrs == self.attributes_count-1: # it is class, ignore it
          continue
        if attrs in final_attr_list:
          acc_list.append(0) # set this attribute as 0, aviod to select
           #if this attribute already in final set
        else:
          temp = []
          for i in final_attr_list:
            temp.append(i)
          temp.append(attrs) # a new attributes list to test, not containing class
          ans = self.kFCV(temp)
          acc_list.append(ans) # insert the accuracy of add this attribute
          print('Accuracy list',acc_list)
      final_attr_list.append(acc_list.index(max(acc_list))) # the max acc index (equal to attribute)
      final_acc_list.append(max(acc_list)) # find the max accuracy
      
      print('Current attributes list',final_attr_list)
      print('Current accuracy list',final_acc_list)
      if len(final_acc_list) >= 2:
        if final_acc_list[-1] < final_acc_list[-2]:
          #final_acc_list = final_acc_list[:-1] # drop the last element, bcuz it's not the best
          print('Final accuracy list，the optimal is located at [-2]',final_acc_list)
          break
    return final_attr_list, final_acc_list

  def BC(self, test, train, att_count): 
    # must have the ability of unsort attrs, which is robust ability
    '''
    process:

    for training data
    1.count the classes type in this data set
    2.calculate the probability of each class and create a dict
    3.for each class, calculate the probability of each attributes, and create a 3 tier dict
    
    for testing data
    1.input a instance 
    2.for each attribute, get the probability from training data
    3.calculate the probability with each class, and predict it as the largest class

    note:
    1. the descreate attributes value from 0 to 9
    2. if its formal train and test now, use D(1,1,1) to D(60,60,60)
    '''
    ##### store the index num of every attrs including class
    index_attrs_class = [] # the index list
    for index in train:
      index_attrs_class.append(index)
    # it will be like [3, 2, 8, 20], 20 is class index, others is attrs we're using now

    ##### start to calculate the class prob
    class_type = [] # count the types of classes in this data set
    class_prob_pair = {} # the prob of each class in the training data
    for class_ in train[index_attrs_class[-1]]: # for each value in class column
      if class_ not in class_type:
        class_type.append(class_)
    for value in train[index_attrs_class[-1]]: # calculate the prob of each class
      if value not in class_prob_pair: # if this key not in dict, create the key and value
        class_prob_pair[value] = 0
      else: # +1
        class_prob_pair[value] += 1
    for key in class_prob_pair.keys():
      class_prob_pair[key] = class_prob_pair[key] / train[index_attrs_class[-1]].count() # the prob of each class
    
    ##### start to calculate the attributes prob
    outter_dict = {} # 3 tier dict: outter, mid, inner
    for class_ in class_type: # for each class
      outter_dict[class_] = {} # mid dict, i.e. outter_dict['Glass_A'] = {}
      filter_ = (train[index_attrs_class[-1]] == class_) # pick the class's instances
      for attr_index in index_attrs_class: # 0 to 3 (or other num in SNB)
        if index_attrs_class.index(attr_index) != len(index_attrs_class)-1: # if it's not 'class'
          outter_dict[class_][attr_index] = {}
          for attr in train[filter_][attr_index]: # for every attributes in this class type
            # for every value in this attributes
            if attr not in outter_dict[class_][attr_index]:
              outter_dict[class_][attr_index][attr] = 1
            else:
              outter_dict[class_][attr_index][attr] += 1 # the count of this value of this attr + 1
              
    for class_2 in outter_dict:
      for attr_2 in outter_dict[class_2]:
        for value in outter_dict[class_2][attr_2]:
          outter_dict[class_2][attr_2][value] = outter_dict[class_2][attr_2][value] / (class_prob_pair[class_2] * train[index_attrs_class[-1]].count())
          
    
    ##### start to input the testing data
    flag_list = [] # if element is 1, means true predict, else false, use it to calculate the accuracy
    for index in range(test.index.start, test.index.stop): # len(df) is the size of this df (num of instance)
      flag = 0 # if this flag == 1, true predict, else false
      predict_2dlist = np.ones((len(class_type), att_count)) # row'a num = class num, column's num = attrs num
      # use ones list is bcuz 1 times every thing is the same
      # att_count not containing class
      ##### start to input the prob of this instance in the 2d list
      for attr_name, attr_value in enumerate(test.loc[index]): # get the values in this instance
        a_n = index_attrs_class[attr_name] # the attribute name, like 3 or 19 or 0, represent "outlook", "wind",...
        if attr_name == att_count:
          continue # means it's class value now
        else:
          if type(attr_value) is type(1) or type(0.1): # avoid missing value
            for first_d_count, class_ in enumerate(outter_dict):
              if attr_value in outter_dict[class_][a_n]:
                predict_2dlist[first_d_count][attr_name] = outter_dict[class_][a_n][attr_value]
          else:
            continue
      ##### start to calculate the prediction of this instance
      product_without_class = []
      for predict_without_class in predict_2dlist:
        product = 1
        for value in predict_without_class:
          product = product * value
        product_without_class.append(product)
      for index_, class_ in enumerate(class_prob_pair.keys()):
        # product times class prob
        product_without_class[index_] = product_without_class[index_] * class_prob_pair[class_]
      # the product_without_class list is product with class now!
      pre_max = max(product_without_class)
      pre_max_class = product_without_class.index(max(product_without_class))
      class_value = test.loc[index].tolist()[-1] # is class value
      for num, class_ in enumerate(class_prob_pair.keys()):
        if class_value == class_:
          if product_without_class[num] == pre_max: # the max value is the true class value
            #right
            flag = 1

      flag_list.append(flag)
    ##### calculate the accuracy of this fold
    acc_temp = flag_list.count(1)
    acc = acc_temp / len(test) 
    return acc

  def kFCV(self, attrslist):
    '''
    process:
    1. create a new df according the given attributes, with class data
    2. do the 5 fold cv and calculate the avg accuracy with self.BC(test, train)
    '''
    #####1
    dflist = {} # create new df first
    for att in attrslist: # through all atributes in the current list (selected)
      templist = [] # the instences values of this attributes list 
      for value in self.df[att]:
        templist.append(value) # append values in the templist
      dflist[att] = templist # dict values given
    
    templist = [] # a new list to store class data
    for class_ in self.df[self.attributes_count-1]: # input the class data into list
      templist.append(class_)
    dflist[self.attributes_count-1] = templist # containing class data in the dict
    current_df = pd.DataFrame(dflist) # create a new df for these attributes with datas

    #####2
    accuracy = []
    for index, value in enumerate(self.k_index):
      if index == 0:
        bc_test_df = current_df[0:value] # 0 to 199
        bc_train_df = current_df[value:self.k_index[self.f_num-1]] # 200 to 999
        acc = self.BC(bc_test_df, bc_train_df, len(attrslist)) # do naive bayesian classification
      elif index != self.f_num-1:
        bc_test_df = current_df[self.k_index[index-1]:value] # 200 to 399
        bc_train_df1 = current_df[0:self.k_index[index-1]] # 0 to 199
        bc_train_df2 = current_df[value:self.k_index[self.f_num-1]] # 400 to 999
        bc_tarin_df = pd.concat([bc_train_df1, bc_train_df2]) # merge two differnet df into 1 df as train data
        acc = self.BC(bc_test_df, bc_train_df, len(attrslist)) # do naive bayesian classification
      else:
        bc_train_df = current_df[0:self.k_index[self.f_num-2]] # 0 to 799
        bc_test_df = current_df[self.k_index[self.f_num-2]:value] # 800 to 999
        acc = self.BC(bc_test_df, bc_train_df, len(attrslist))
      accuracy.append(acc) # append accuracy in the list
    ans = sum(accuracy)/self.f_num # avg
    return ans # accuracy of 5 fold's avg

  def see_attrs_code_pair(self):
    print(self.attributes_code_pairs)
