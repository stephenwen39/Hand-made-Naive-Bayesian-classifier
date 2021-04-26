class NBC(object):

  def __init__(self, cropus_root="/content/gdrive/My Drive/Colab Notebooks/Naive Bayesian Classifier/data/", file_="train.csv"):
    #You should change it to your own path 
    self.cropus_root = cropus_root
    self.df = pd.read_csv(self.cropus_root + file_)
    
  def encoder(self, df_column): #plz input a discrete attribute column, not all dataframe
    '''
    encoder can return an integer list based on attributes value, for example, input[male, male, female], output[0,0,1]
    '''
    new_attributes = []
    value_code_pair = {}
    count = 0
    for i in self.df[df_column]:
        if i not in value_code_pair:
            value_code_pair[i] = count
            count += 1
        else:
            continue
    for index, value in enumerate(self.df[df_column]):
        if value in value_code_pair: #if the value already in dict
            new_attributes.insert(index, value_code_pair[value]) #replace the original value with code
    return new_attributes #return a list of code
    '''
    REMIND: put a name of column, which is string type
    e.g.
    encoder('Sex')
    '''

  def hybird_discret(self, name_of_attribute, class_data): #input hybird_discret(df['Age'],df['Survived'])
    '''
    hybird_discret can help you to count the ri value of your single attribute(based on paper
    "Wong, T. T. (2012). A hybrid discretization method for naïve Bayesian classifiers. Pattern Recognition, 45(6), 2321-2325."
    ri means the times class type jump between the same class. 
    '''
    self.new_df = self.df.sort_values(by=[name_of_attribute]) #sort by the attribute first
    print(self.new_df['Survived'])
    atts_list = [] #the container of attribute and class, which will process later
    class_list = []
    
    for atts, class_ in zip(self.new_df[name_of_attribute], self.new_df[class_data]):
      atts_list.append(atts) #input to new list to process it
      class_list.append(class_)
    
    in_flag = 0
    out_flag = 0
    jump = 0
    jump_time = [] #count number of jump times for every class type
    
    for class_type in range(0,len(set(class_list))): #the number of different class
      in_flag = 0
      out_flag = 0
      jump = 0
      
      for index, value in enumerate(class_list): #class need to encode in integer
        if value == class_type:
          if out_flag == 0: #means first time to see this class
            in_flag = 1
          elif out_flag == 1: #means count should ++, which means jump times ++
            jump += 1
            out_flag = 0 #pointer inside now
            in_flag = 1
        else:
          if index != 0: #to prevent first element error
            if class_list[index - 1] == class_type: #leave the class area
              out_flag = 1
              in_flag = 0
      jump_time.append(jump)
      
    return jump_time #return the jump time per every element in this single attribute
