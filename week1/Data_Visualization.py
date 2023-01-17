
'''
Matplotlib: It is the most used library for plotting in the Python community, despite having more than a decade of development. Because matplotlib was the first Python data visualization library, many other libraries are built on top of it. Some libraries like pandas and Seaborn are wrappers over matplotlib.
Seaborn: leverages matplotlib's ability to create aesthetic graphics in a few lines of code. The most palpable difference is Seaborn's default styles and color palettes, which are designed to be more aesthetically pleasing and modern.
Pandas plotting: allows data visualization through adaptations of the matplotlib library, facilitating the data aggregation and manipulation in a few lines of code.
Plotly: allows the data viusalization by interactive plots, offering additional chart configurations as contour plots, dendograms, and 3D charts.
ggplot: is based on ggplot2 from R plotting system. ggplot operates differently than matplotlib and seaborn, making layers fromo its components to create a complete plot.
Bokeh: creates interactive, web-ready plots, which can be easily output as JSON objects, HTML documents, or interactive web applications, supporting streaming and real-time data.
AstroPy: is a collection of software packages written in the Python, and designed for use in astronomy.
Gleam: is inspired by R's Shiny package. It allows to turn analysis into interactive web applications using only Python scripts, avoiding the use of other languages like HTML, CSS, or JavaScript.
Geoplotlib: is a toolbox for creating maps and plotting geographical data by creating a variety of map-types, like choropleths, heatmaps, and dot density maps.
Missingno: allows to quickly gauge the completeness of a dataset with a visual summary, instead of trudging through a table.'''


# 1. Matplotlib
# matplotlib.pyplot is a collection of command style functions that make matplotlib work like MATLAB.
'''
Matplotlib charts are made up of two main components:

The axes: the lines that delimit the area of the chart
The figure: where we draw the axes, titles and elements that come out of the area of the axes.'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1.3. Loading Dataset
x = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/Standard_Metropolitan_Areas_Data-data.csv") #let's load a dataset

x.head() #check what all variables/fields are there in the dataset

# 1.4. Scatter Plot using Matplotlib

plt.scatter(x.crime_rate, x.percent_senior) # Plotting the scatter plot
# Here we are creating a scatter plot between crime rate and percent senior variables
plt.show() # Showing the figure

'''
Scatter plots are used to observe relationships between variables.
We can divide data points into groups based on how closely sets of points cluster together.
Scatter plots can also show if there are any unexpected gaps in the data and if there are any outlier points.(Look at the 2 points away from rest of the data in the scatter plot. Those are outliers.)
'''

'''
Is plt.show() always required?
If Matplotlib is used in a terminal, scripts or specialized IDEs as Spyder, Pycharm or VS Code, plt.show() is a must.

If Matplotlib is used in a IPython shell or a notebook as Jupyter Notebook or Colab Notebook, plt.show() is usually unnecessary.'''

# The same code block without plt.show() gives the same result in Jupyter Notebook
plt.scatter(x.crime_rate, x.percent_senior)

# Adding titles and labels
plt.scatter(x.percent_senior, x.crime_rate)

plt.title('Plot of Crime Rate vs Percent Senior') # Adding a title to the plot
plt.xlabel("Percent Senior") # Adding the label for the horizontal axis
plt.ylabel("Crime Rate") # Adding the label for the vertical axis
plt.show()

# 1.5. Line Chart using Matplotlib

plt.plot(x.work_force, x.income) # 2 arguments: X and Y points
plt.xlabel("Work Force") # Adding the label for the horizontal axis
plt.ylabel("Income")
plt.show()

plt.plot([1, 2, 3, 4]) # 1 argument
plt.show()

# Changing the size of the plot

plt.figure(figsize=(12,5)) # 12x5 plot

plt.plot(x.work_force, x.income)
plt.xlabel("Work Force")
plt.ylabel("Income")
plt.show()

# Formatting the style of your plot
plt.plot(x.work_force, x.income, linestyle='--', marker='o', color='r')
plt.xlabel("Work Force")
plt.ylabel("Income")
plt.show()

# A shortcut for the above
# The default format string is 'b-', which is a solid blue line.
plt.plot(x.work_force, x.income, '--ro') # ro = red circles
plt.xlabel("Work Force")
plt.ylabel("Income")
plt.show()

plt.plot(x.work_force, x.income, 'ro') # ro = red circles
plt.show()

plt.plot(x.work_force, x.income, "gx") # gx = green x
plt.show()

'''
plt.plot(x.work_force, x.income, "go") # green circles
plt.plot(x.work_force, x.income, "g^") # green traingles
plt.plot(x.work_force, x.income, "ro") # red circles
plt.plot(x.work_force, x.income, "rx") # red x symbol
plt.plot(x.work_force, x.income, "b^") # red ^ symbol
plt.plot(x.work_force, x.income, "go--", linewidth=3) # green circles and dashed lines of width 3.
'''
# 1.6. Plotting consecutive plots using Matplotlib

plt.plot(x.work_force, x.income, color="r")
plt.plot(x.physicians, x.income)
plt.show()

# if you don't use plt.show() after the first figure? Both variables will be plotted in the same figure


# Adding a Legend
# (both share the same axes)
plt.plot(x.work_force, x.income, color="r", label = 'work_force')
plt.plot(x.physicians, x.income, label='physicians')

# Adding a legend
plt.legend()

plt.show()

# 1.7. Multiple plots in one figure using Matplotlib

'''
The .subplot() method is used to add multiple plots in one figure. It takes three arguments:

nrows: number of rows in the figure
ncols: number of columns in the figure
index: index of the plot'''

# 1 row and 2 columns
plt.subplot(1,2,1) # row, column, index
plt.plot(x.work_force, x.income, "go")
plt.title("Income vs Work Force")

## plt.subplot(1,2,2) # row, column, index
plt.subplot(1,2,2).label_outer()
plt.plot(x.hospital_beds, x.income, "r^")
plt.title("Income vs Hospital Beds")

plt.suptitle("Sub Plots") # Add a centered title to the figure.
plt.show()

# 2 rows and 1 column
plt.subplot(2,1,1) # row, column, index
plt.plot(x.work_force, x.income, "go")

plt.subplot(2,1,2) # row, column, index
plt.plot(x.hospital_beds, x.income, "r^")

plt.suptitle("Sub Plots")
plt.show()

# 2 rows and 2 columns

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(6,6)) #creating a grid of 2 rows, 2 columns and 6x6 figure size
ax[0,0].plot(x.work_force, x.income, "go") # The top-left axes
ax[0,1].plot(x.work_force, x.income, "bo") # The top-right axes
ax[1,0].plot(x.work_force, x.income, "yo") # The bottom-left axes
ax[1,1].plot(x.work_force, x.income, "ro") # The bottom-right axes

plt.show()

# 1.8. Histogram using Matplotlib
'''
arameters:
x(n,) : this takes either a single array or a sequence of arrays which are not required to be of the same length.
bins : If bins is an integer, it defines the number of equal-width bins in the range.'''

plt.title("Histogram")
plt.xlabel("Percentage of Senior Citizens")
plt.ylabel("Frequency")

plt.hist(x.percent_senior)
plt.show()

# 2. Seaborn
# 2.2. Loading data with seaborn
import seaborn as sns
# Load iris data
iris = sns.load_dataset("iris")

iris.sample(10)

# 2.3. Scatter plot using seaborn
sns.scatterplot(x="sepal_length", y="sepal_width", data=iris)
plt.show()

# 2.4. Swarm Plot using Seaborn

# Construct swarm plot for sepcies vs petal_length
sns.swarmplot(x="species", y="petal_length", data=iris)

# Show plot
plt.show()

# 2.5. Heatmap using Seaborn
#　correlation matrix visualization ( exploring the correlation of features in a dataset)

# Correlation matrix completely ignores any non-numeric column.
sns.heatmap(iris.corr(), annot=True)
plt.show()

# Vertical Bar Charts
plt.bar(x.region, x.crime_rate, color="green")

plt.title("Bar Graph")
plt.xlabel("Region")
plt.ylabel("Crime Rate")
plt.show()

# Horizontal Bar Charts
plt.barh(x.region, x.crime_rate, color="green")
plt.title("Bar Graph")
plt.show()

# Bar Charts with multiple quantities
divisions = ["A", "B", "C", "D", "E"]
division_avg = [70, 82, 73, 65, 68]
boys_avg = [68, 67, 77, 61, 70]

# Using the NumPy arange function to generate values for index between 0-4.
# Here,stop is 5, start is 0, and step is 1
index = np.arange(5)
width = 0.30

plt.bar(index, division_avg, width, color="green", label="Division Marks")
plt.bar(index+width, boys_avg, width, color="blue", label="Boys Marks")

plt.title("Bar Graph")
plt.xlabel("Divisions")
plt.ylabel("Marks")
plt.show()

# Stacked Bar Chart
divisions = ["A", "B", "C", "D", "E"]
girls_avg = [72, 97, 69, 69, 66]
boys_avg = [68, 67, 77, 61, 70]

index = np.arange(5)
width = 0.50

plt.bar(index, boys_avg, width, color="green", label="Boys Marks")
plt.bar(index, girls_avg, width, color="blue", label="Girls Marks", bottom=boys_avg)

plt.title("Bar Graph")
plt.xlabel("Divisions")
plt.ylabel("Marks")
plt.show()

# 3.2. Pie Chart using Matplotlib

'''Parameters of a pie chart:

x: array-like. The wedge sizes.
labels: list. A sequence of strings providing the labels for each wedge.
Colors: A sequence of matplotlibcolorargs through which the pie chart will cycle. If None, will use the colors in the currently active cycle.
Autopct: string, used to label the wedges with their numeric value. The label will be placed inside the wedge. The format string will be fmt%pct.
'''
firms = ["Firm A", "Firm B", "Firm C", "Firm D", "Firm E"]
market_share = [20,25,15,10,20]

# Explode the pie chart to emphasize a certain part or some parts( Firm B in this case)
# It is useful because it makes the highlighted portion more visible.
Explode = [0,0.1,0,0,0]

plt.pie(market_share, explode=Explode, labels=firms, autopct='%1.1f%%', startangle=45)

plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.

plt.legend(title="List of Firms")

plt.show()



