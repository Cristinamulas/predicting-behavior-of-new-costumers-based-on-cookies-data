import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def value__counts(df_):
    '''it takes a df and return value_counts for all the features'''
    for i in df_.columns:
        print(df_[i].value_counts())
        
        
def count_freq_plot(df, col,name, want_percentages = False):
    " it retuns a bar plot with the percentages or w/o percentages"
    
    sns.set(style="darkgrid")
    plt.figure(figsize=(8,6))
    total = float(len(df)) 
    ax = sns.countplot(x=col, data=df, dodge =False) # for Seaborn version 0.7 and more
    ax.set(xlabel=col, title= f'Count Frequency: Positive {name} vs Negative {name} ')


    if want_percentages == True:
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x()+p.get_width()/2., height + 3,'{:1.2%}'.format(height/total),ha="center") 

        plt.show()