

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data
# week 0 (and 1 quiz from week 1)
df = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/DPhi%20-%20Learners%20-%20Beginners%20%26%20Absolute%20Beginners%20-%20Real%20Dataset%20-%20DPhi_Learners.csv")
df.info()
df.head()

# Transform Data

# filter for ab_g1
df = df[df.Group_ID == "AB_G1"]
# missing values were entered as "-" characters. This is also the reason why all dtypes were imported as "object". Replacing them with numpy's NaN solves this and we now have more appropriate dtypes.
# replace "-" by NAN
df = df.replace("-", np.nan)
df.info()
df.head()

# df["Total_Score"] = pd.to_numeric(df["Total_Score"])
# df["Quiz1"] = pd.to_numeric(df["Quiz1"])

cols = df.columns.drop(['Learner_ID', 'Group_ID', 'Learner_Category'])
df[cols] = df[cols].apply(pd.to_numeric)
df.info()

# get leader list
leaderboard_list = (df.loc[df["Total_Score"] > 0, ["Learner_ID", "Total_Score"]]
.sort_values("Total_Score", ascending=False)
.reset_index(drop=True))
print(leaderboard_list)

# heatmap df
df_heatmap = (df[df["Total_Score"] > 0]
.sort_values("Total_Score", ascending=False)
.drop(["Total_Score", "Group_ID", "Learner_Category"], axis=1)
.set_index("Learner_ID"))
df_heatmap.head()

# Visualization
'''
For the heatmap, we select the colormap to be "BuGn" and specify that NaN values should get the color "lightgray". Then we set up the plot using sns.heatmap() using the df we created earlier and select that the numbers should also be included in the rectangles by setting annot=True.
Also remember that if we use the object oriented API of matplotlib, we have to use the "ax" object to make changes to the plot. In the seaborn call, we also have to select on which of the two axes of the subplot we want to draw the graph. In our case, the left one is ax[0] and the right one is ax[1]. We then rotate the ticks to make them more readable and set a title.
Finally, we also create the barchart that displays the total counts. We add annotations containing the total scores next to the bars by iterating over the ax object. If this looks complicated to you, remember that you can just copy it for your own plots for now and come back to it once you have explored the library more. To avoid labels that are cut-off, we increase the limits on the x-axis.'''

# initialize plot
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
# heatmap
plt.cm.get_cmap(name="BuGn").set_bad("lightgray")
heatmap = sns.heatmap(df_heatmap, cmap="BuGn", annot=True, linewidths=.5, ax=ax[0])
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45)
ax[0].set_title("Score per Quiz")
# bar chart
sns.barplot(x="Total_Score", y="Learner_ID", data=leaderboard_list, ax=ax[1])
ax[1].set_xlim(0, 70)
ax[1].set_title("Total Score")
for p in ax[1].patches:
    ax[1].annotate("{0:.2f}".format(p.get_width()),
    xy=(p.get_width(), p.get_y() + p.get_height() / 2),
    xytext=(5, 0),
    textcoords='offset points', ha="left", va="center")
plt.show()