# Welcome to our [team project website!](https://julioveracruz.github.io/testwebsite/)

This website showcases our final project for FIN 377 - Data Science for Finance course at Lehigh University.

To see our complete analysis file, click [here](https://github.com/julioveracruz/testwebsite/blob/main/notebooks/example.ipynb).


## Table of contents
1. [Introduction](#introduction)
2. [Methodology](#meth)
3. [Section 2](#section2)
    1. [Subsection](#subsec2-1)
    2. [Subsection](#subsec2-2)
4. [Analysis Section](#section3)
5. [Summary](#summary)

## Introduction  <a name="introduction"></a>

The main goal of this project is to explore efficient machine learning models that can best predict loan defaults using data such as standard loan factors, macroeconomic data, and market conditions.  

## Methodology <a name="meth"></a>

Here is some code that we used to develop our analysis. We outlined and completed this project through steps of:

- Data Collection
- EDA
- Preprocessing
- Optimizing
- Model Selection
 
Note that for the purposes of the website, you have to copy this code into the markdown file and  
put the code inside trip backticks with the keyword `python`.

```python
import seaborn as sns 
iris = sns.load_dataset('iris') 

print(iris.head(),  '\n---')
print(iris.tail(),  '\n---')
print(iris.columns, '\n---')
print("The shape is: ",iris.shape, '\n---')
print("Info:",iris.info(), '\n---') # memory usage, name, dtype, and # of non-null obs (--> # of missing obs) per variable
print(iris.describe(), '\n---') # summary stats, and you can customize the list!
print(iris['species'].value_counts()[:10], '\n---')
print(iris['species'].nunique(), '\n---')
```

Notice that the output does NOT show! **You have to copy in figures and tables from the notebooks.**

## Section <a name="section2"></a>
Blah blah

### Subsection 1 <a name="subsec2-1"></a>
This is a subsection, formatted in heading 3 style

### Subsection 2 <a name="subsec2-2"></a>
This is a subsection, formatted in heading 3 style

## Analysis Section <a name="section3"></a>

Here are some graphs that we created in our analysis. We saved them to the `pics/` subfolder and include them via the usual markdown syntax for pictures.

![](pics/plot1.png)
<br><br>
Some analysis here
<br><br>
![](pics/plot2.png)
<br><br>
More analysis here.
<br><br>
![](pics/plot3.png)
<br><br>
More analysis.

## Summary <a name="summary"></a>

Blah blah



## About the team

<img src="pics/julio.jpg" alt="julio" width="300"/>
<br>
Eric is a senior studying Finance.
<br><br><br>
<img src="pics/don2.jpg" alt="don" width="300"/>
<br>



## More 

To view the GitHub repo for this website, click [here](https://github.com/donbowen/teamproject).
