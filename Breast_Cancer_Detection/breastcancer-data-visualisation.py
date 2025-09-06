# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Download latest version
path = 'Breast_Cancer_Detection'
data = pd.read_csv(f"{path}/breast-cancer-dataset.csv")


# Set the display.max_columns option to None
pd.set_option('display.max_columns', None)
# Vieweing 5 first data
print(data.head())
# Viewing 5 latest data
print(data.tail())
print(data.info())
print(data.shape)

df = pd.DataFrame(data, columns=['Year', 'Age','Menopause','Tumor Size (cm)','Inv-Nodes','Breast','Metastasis','Breast Quadrant','History','Diagnosis Result'])

#finding unique values

print('Age',df['Age'].unique())
print('Year',df['Year'].unique())
print('Menopause',df['Menopause'].unique())
print('Tumor Size (cm)',df['Tumor Size (cm)'].unique())
print('Inv-Nodes',df['Inv-Nodes'].unique())
print('Breast',df['Breast'].unique())
print('Metastasis',df['Metastasis'].unique())
print('Breast Quadrant',df['Breast Quadrant'].unique())
print('History',df['History'].unique())
print('Diagnosis Result',df['Diagnosis Result'].unique())

#finding missing values (#)

print('Age # Indexes',df[df['Year']=='#'].index.values)
print('Tumor Size (cm) # Indexes',df[df['Tumor Size (cm)']=='#'].index.values)
print('Inv-Nodes # Indexes',df[df['Inv-Nodes']=='#'].index.values)
print('Metastasis # Indexes',df[df['Metastasis']=='#'].index.values)
print('Breast # Indexes',df[df['Breast']=='#'].index.values)
print('Metasis # Indexes',df[df['Metastasis']=='#'].index.values)
print('Breast Quadrant # Indexes',df[df['Breast Quadrant']=='#'].index.values)
print('History # Indexes',df[df['History']=='#'].index.values)

# Clean the dataset by removing rows with missing values
dataset_cleaned = df.copy()
dataset_cleaned= dataset_cleaned.drop([30,40,47,67,143,164,166,178,208])

#Descriptive statistics
def describe(df):

  features=[]
  dtypes=[]
  count=[]
  unique=[]
  missing_values=[]
  min_ =[]
  max_ =[]

  for item in df.columns:
    features.append(item)
    dtypes.append(df[item].dtype)
    count.append(len(df[item]))
    unique.append(len(df[item].unique()))
    missing_values.append(df[item].isna().sum())

    if df[item].dtypes == 'int64' or df[item].dtypes == 'float64':

      min_.append(df[item].min())
      max_.append(df[item].max())

    else:
      min_.append('NaN')
      max_.append('NaN')

  out_put = pd.DataFrame({'Feature':features,'Dtype':dtypes,'Count':count,'Unique':unique,'Missing_value':missing_values,
                          'Min':min_,'Max':max_})

  return out_put.T

print(describe(df))
print(dataset_cleaned.shape)
#numerise the data
dataset = dataset_cleaned.copy()
dataset['Tumor Size (cm)'] = pd.to_numeric(dataset['Tumor Size (cm)']) 
dataset['Inv-Nodes'] = pd.to_numeric(dataset['Inv-Nodes'])
dataset['Metastasis'] = pd.to_numeric(dataset['Metastasis'])
dataset['History'] = pd.to_numeric(dataset['History'])

df_names =dataset.copy()
df_names['Menopause'] = df_names['Menopause'].replace(1,'Yes')
df_names['Menopause'] = df_names['Menopause'].replace(0,'No')
df_names['History'] = df_names['History'].replace(1,'Yes')
df_names['History'] = df_names['History'].replace(0,'No')
df_names['Inv-Nodes'] = df_names['Inv-Nodes'].replace(1,'Yes')
df_names['Inv-Nodes'] = df_names['Inv-Nodes'].replace(0,'No')
df_names['Metastasis'] = df_names['Metastasis'].replace(1,'Yes')
df_names['Metastasis'] = df_names['Metastasis'].replace(0,'No')


#Change malignant to 1 and benign to 0
dataset['Diagnosis Result'] = dataset['Diagnosis Result'].replace('Malignant',1)
dataset['Diagnosis Result'] = dataset['Diagnosis Result'].replace('Benign',0)
#Change Breat left to 2 and right to 1
dataset['Breast'] = dataset['Breast'].replace('Left',2)
dataset['Breast'] = dataset['Breast'].replace('Right',1)
#Change Breast Quadrant to 1,2,3,4
dataset['Breast Quadrant'] = dataset['Breast Quadrant'].replace('Upper inner',1)
dataset['Breast Quadrant'] = dataset['Breast Quadrant'].replace('Upper outer',2)
dataset['Breast Quadrant'] = dataset['Breast Quadrant'].replace('Lower inner',3)
dataset['Breast Quadrant'] = dataset['Breast Quadrant'].replace('Lower outer',4)


import matplotlib.pyplot as plt
data = ['Age', 'Menopause', 'Tumor Size (cm)', 'Inv-Nodes', 'Breast','Metastasis', 'Breast Quadrant', 'History']


healthy_num = dataset[dataset['Diagnosis Result']== 0]
df_healthy_num = pd.DataFrame(healthy_num)

disease_num = dataset[dataset['Diagnosis Result']==1]
df_disease_num = pd.DataFrame(disease_num)

healthy = df_names[df_names['Diagnosis Result']== 'Benign']
df_healthy = pd.DataFrame(healthy)

disease = df_names[df_names['Diagnosis Result']=='Malignant']
df_disease = pd.DataFrame(disease)


# Print out age vs Diagnosis Result
fig,axes = plt.subplots(nrows = 2, ncols = 1, figsize = (15,8),  facecolor='#f2faf9',dpi = 150)

fig.suptitle('Age of healthy women vs  women having cancer ',fontsize=20)

ax=sns.countplot(x =df_healthy_num['Age'],palette="viridis",ax = axes[0])
ax.set_xlabel('Age',fontsize=17)
ax.set_ylabel('Count',fontsize= 17)
ax.set_title("Benign", fontsize =20, y= 1.05)

for p in ax.patches:

  ax.annotate(format(p.get_height()),
                    (p.get_x()+ p.get_width()/2,
                      p.get_height()), ha='center', va='center',
                    size=8,color='white',
                    weight='bold', backgroundcolor='black', xytext=(0, 10),
                    textcoords='offset points', bbox=dict(boxstyle='round,pad=0.3',
                    edgecolor='white', lw=0))


ax=sns.countplot(x = df_disease_num['Age'], palette = 'gist_heat',ax = axes[1])

ax.set_xlabel('Age',fontsize=17)
ax.set_ylabel('Count',fontsize= 17)
ax.set_title("Malignant",fontsize = 20, y= 1.05)

for p in ax.patches:

  ax.annotate(format(p.get_height()),
                    (p.get_x()+ p.get_width()/2,
                      p.get_height()), ha='center', va='center',
                    size=8,color='white',
                    weight='bold', backgroundcolor='black', xytext=(0, 10),
                    textcoords='offset points', bbox=dict(boxstyle='round,pad=0.3',
                    edgecolor='white', lw=0))


plt.tight_layout(pad =2.0)
plt.savefig('Breast_Cancer_Detection/Results/Age_vs_cancerdiagnosis.png')



# Sort by age
df_age = df_names.copy()
df_age = df_age.sort_values(by=["Age"])
df_age = df_age.reset_index()
df_age.drop(columns=["index"],inplace=True)


plt.figure(figsize=(5,4))
sns.displot(data=df_age, x="Age",hue='Diagnosis Result',rug=True)
plt.title("age vs Diagnosis Result", y = 1.05)

plt.savefig('Breast_Cancer_Detection/Results/sortedAge_vs_cancerdiagnosis.png')

# Tumor Size

plt.figure(figsize=(8,5), facecolor='#f2faf9')
sns.kdeplot(df_healthy_num['Tumor Size (cm)'],shade =True )
sns.kdeplot(df_disease_num['Tumor Size (cm)'],shade =True)
plt.legend(['Benign','Malignant'])
plt.title('Tumor Size in healthy women vs women having breast cancer' ,y=1.05)

plt.savefig('Breast_Cancer_Detection/Results/TumorSize_vs_cancerdiagnosis.png')

# Menopause
plt.figure(figsize=(8,5), facecolor='#f2faf9')
sns.kdeplot(df_healthy_num['Menopause'],shade =True )
sns.kdeplot(df_disease_num['Menopause'],shade =True)
plt.legend(['Benign','Malignant'])
plt.title('Menopause in healthy women vs women having breast cancer ' ,y=1.05)

plt.savefig('Breast_Cancer_Detection/Results/menopause_vs_cancerdiagnosis.png')

#Inv Nodes 
plt.figure(figsize=(8,5), facecolor='#f2faf9')
sns.kdeplot(df_healthy_num['Inv-Nodes'],shade =True )
sns.kdeplot(df_disease_num['Inv-Nodes'],shade =True)
plt.legend(['Benign','Malignant'])
plt.title('The axillary lymph nodes containing metastatic in healthy women vs  women having cancer ' ,y=1.05)

plt.savefig('Breast_Cancer_Detection/Results/Invnodes_vs_cancerdiagnosis.png')

# Breast Side
fig,axes = plt.subplots(nrows = 2, ncols = 1, figsize = (8,4),  facecolor='#f2faf9',dpi = 90)

fig.suptitle('Breast side in healthy women vs women having breast cancer ')

ax=sns.countplot(x =df_healthy['Breast'],palette="viridis",width=0.2,ax = axes[0])
ax.set_title("Benign",fontsize=15,y=1.05)

for p in ax.patches:

  ax.annotate(format(p.get_height()),
                    (p.get_x()+ p.get_width()/2,
                      p.get_height()), ha='center', va='center',
                    size=8,color='white',
                    weight='bold', backgroundcolor='black', xytext=(0, 5),
                    textcoords='offset points', bbox=dict(boxstyle='round,pad=0.3',
                    edgecolor='white', facecolor='black', lw=0))

ax=sns.countplot(x = df_disease['Breast'], palette = 'gist_heat',width=0.2,ax = axes[1])
ax.set_title("Malignant",fontsize=15,y=1.05)
for p in ax.patches:

  ax.annotate(format(p.get_height()),
                    (p.get_x()+ p.get_width()/2,
                      p.get_height()), ha='center', va='center',
                    size=8,color='white',
                    weight='bold', backgroundcolor='black', xytext=(0, 5),
                    textcoords='offset points', bbox=dict(boxstyle='round,pad=0.3',
                    edgecolor='white', facecolor='black', lw=0))

plt.tight_layout(pad=2.0)
plt.savefig('Breast_Cancer_Detection/Results/breastside_vs_cancerdiagnosis.png')

#Metastsis

plt.figure(figsize=(7,5), facecolor='#f2faf9')
sns.kdeplot(df_healthy_num['Metastasis'],shade =True )
sns.kdeplot(df_disease_num['Metastasis'],shade =True)
plt.title('Metastasis in healthy women vs women having breast cancer', y=1.05)
plt.legend(['Benign','Malignant'])
plt.savefig('Breast_Cancer_Detection/Results/metastasis_vs_cancerdiagnosis.png')

# Breast Quadrant

fig,axes = plt.subplots(nrows = 2, ncols = 1, figsize = (8,4),  facecolor='#f2faf9',dpi = 90)

fig.suptitle('Breast Quadrant in healthy women and women having breast cancer')

ax= sns.countplot(x =df_healthy['Breast Quadrant'],palette="viridis",width= 0.2,ax = axes[0])
ax.set_title("Benign",fontsize=15,y=1.05)

for p in ax.patches:

  ax.annotate(format(p.get_height()),
                    (p.get_x()+ p.get_width()/2,
                      p.get_height()), ha='center', va='center',
                    size=8,color='white',
                    weight='bold', backgroundcolor='black', xytext=(0, 5),
                    textcoords='offset points', bbox=dict(boxstyle='round,pad=0.3',
                    edgecolor='white', facecolor='black', lw=0))

ax = sns.countplot(x = df_disease['Breast Quadrant'], palette = 'gist_heat',width= 0.2,ax = axes[1])
ax.set_title("Malignant",fontsize=15,y =1.05)
for p in ax.patches:

  ax.annotate(format(p.get_height()),
                    (p.get_x()+ p.get_width()/2,
                      p.get_height()), ha='center', va='center',
                    size=8,color='white',
                    weight='bold', backgroundcolor='black', xytext=(0, 5),
                    textcoords='offset points', bbox=dict(boxstyle='round,pad=0.3',
                    edgecolor='white', facecolor='black', lw=0))

plt.tight_layout(pad=2.0)
plt.savefig('Breast_Cancer_Detection/Results/cancerquadrant_vs_cancerdiagnosis.png')

#History

plt.figure(figsize=(8,5), facecolor='#f2faf9')
sns.kdeplot(df_healthy_num['History'],shade =True )
sns.kdeplot(df_disease_num['History'],shade =True)
plt.legend(['Benign','Malignant'])
plt.title('History in healthy women vs women having breast cancer ' ,y=1.05)


plt.savefig('Breast_Cancer_Detection/Results/history_vs_cancerdiagnosis.png')


# Pie Charts
#healthy Count
age_count=df_healthy["Age"].value_counts()
Menopause_count = df_healthy['Menopause'].value_counts()
Tumor_Size_count = df_healthy['Tumor Size (cm)'].value_counts()
Inv_Nodes_count = df_healthy['Inv-Nodes'].value_counts()
Breast_count=df_healthy['Breast'].value_counts()
Metastasis_count = df_healthy['Metastasis'].value_counts()
Breast_Quadrant_count = df_healthy['Breast Quadrant'].value_counts()
History_count = df_healthy['History'].value_counts()
#Healthy Count
age_uniq=df_healthy["Age"].unique()
Menopause_uniq = df_healthy['Menopause'].unique()
Tumor_Size_uniq = df_healthy['Tumor Size (cm)'].unique()
Inv_Nodes_uniq = df_healthy['Inv-Nodes'].unique()
Breast_uniq=df_healthy['Breast'].unique()
Metastasis_uniq = df_healthy['Metastasis'].unique()
Breast_Quadrant_uniq = df_healthy['Breast Quadrant'].unique()
History_uniq = df_healthy['History'].unique()
#Disease Count
age_disease_count=df_disease["Age"].value_counts()
Menopause_disease_count = df_disease['Menopause'].value_counts()
Tumor_Size_disease_count = df_disease['Tumor Size (cm)'].value_counts()
Inv_Nodes_disease_count = df_disease['Inv-Nodes'].value_counts()
Breast_disease_count=df_disease['Breast'].value_counts()
Metastasis_disease_count = df_disease['Metastasis'].value_counts()
Breast_Quadrant_disease_count = df_disease['Breast Quadrant'].value_counts()
History_disease_count = df_disease['History'].value_counts()
#Disease Unique
age_disease_uniq=df_disease["Age"].unique()
Menopause_disease_uniq = df_disease['Menopause'].unique()
Tumor_Size_disease_uniq = df_disease['Tumor Size (cm)'].unique()
Inv_Nodes_disease_uniq = df_disease['Inv-Nodes'].unique()
Breast_disease_uniq=df_disease['Breast'].unique()
Metastasis_disease_uniq = df_disease['Metastasis'].unique()
Breast_Quadrant_disease_uniq = df_disease['Breast Quadrant'].unique()
History_disease_uniq = df_disease['History'].unique()

#Menopause Pie Chart

palette_color = sns.color_palette("pastel")

# Wedge properties
wp = {'linewidth': 1, 'edgecolor': "green"}

# Creating autocpt arguments
explode = (0.1, 0.0 )

def func(pct, allvalues):
    absolute = int(pct / 100.*np.sum(allvalues))
    return "{:.1f}%".format(pct, absolute)

fig,ax = plt.subplots(nrows=1, ncols =2 , figsize=(7,7),dpi=100)

fig.suptitle("Menopause vs bresat cancer")

#################################### Benign #################################

wedges, texts, autotexts = ax[0].pie(Menopause_count,
                                  autopct=lambda pct: func(pct, Menopause_count),

                                  labels=Menopause_uniq,
                                  shadow=True,
                                  explode =explode,
                                  colors=palette_color,
                                  startangle=90,
                                  wedgeprops=wp,
                                  textprops=dict(color="steelblue"))


plt.setp(autotexts, size=9, weight="bold")
ax[0].set_title("Menopause and Benign Tumor",pad=60)
ax[0].legend(title="Menopause", loc="best")


################################# malignant #################################

wedges, texts, autotexts = ax[1].pie(Menopause_disease_count,
                                  autopct=lambda pct: func(pct, Menopause_disease_count),

                                  labels=Menopause_disease_uniq,
                                  shadow=True,
                                  explode =explode,
                                  colors=palette_color,
                                  startangle=90,
                                  wedgeprops=wp,
                                  textprops=dict(color="steelblue"))


plt.setp(autotexts, size=9, weight="bold")
ax[1].set_title("Menopause and Malignant Tumor",pad=60)
ax[1].legend(title="Menopause",loc = "best")

plt.tight_layout()
plt.savefig('Breast_Cancer_Detection/Results/Menopause_piechart.png')

#Breast Side Pie chart

palette_color = sns.color_palette("pastel")

# Wedge properties
wp = {'linewidth': 1, 'edgecolor': "green"}

# Creating autocpt arguments
explode = (0.1, 0.0 )

def func(pct, allvalues):
    absolute = int(pct / 100.*np.sum(allvalues))
    return "{:.1f}%".format(pct, absolute)

fig,ax = plt.subplots(nrows=1, ncols =2 , figsize=(7,7),dpi=100)

fig.suptitle("Breast vs bresat cancer")

#################################### Benign #################################

wedges, texts, autotexts = ax[0].pie(Breast_count,
                                  autopct=lambda pct: func(pct, Breast_count),

                                  labels=Breast_uniq,
                                  shadow=True,
                                  explode =explode,
                                  colors=palette_color,
                                  startangle=90,
                                  wedgeprops=wp,
                                  textprops=dict(color="steelblue"))


plt.setp(autotexts, size=9, weight="bold")
ax[0].set_title("Breast and Benign Tumor",pad=60)
ax[0].legend(title="Breast")

################################# malignant #################################

wedges, texts, autotexts = ax[1].pie(Breast_disease_count,
                                  autopct=lambda pct: func(pct, Breast_disease_count),

                                  labels=Breast_disease_uniq,
                                  shadow=True,
                                  explode =explode,
                                  colors=palette_color,
                                  startangle=90,
                                  wedgeprops=wp,
                                  textprops=dict(color="steelblue"))


plt.setp(autotexts, size=9, weight="bold")
ax[1].set_title("Breast and Malignant Tumor",pad=60)
ax[1].legend(title="Breast")

plt.tight_layout()
plt.savefig('Breast_Cancer_Detection/Results/Breast_piechart.png')

## Breast Quadrant Pie chart

palette_color = sns.color_palette("pastel")

# Wedge properties
wp = {'linewidth': 1, 'edgecolor': "green"}

# Creating autocpt arguments
explode = (0.1, 0.2,0.0,0.0 )

def func(pct, allvalues):
    absolute = int(pct / 100.*np.sum(allvalues))
    return "{:.1f}%".format(pct, absolute)

fig,ax = plt.subplots(nrows=1, ncols =2 , figsize=(12,12),dpi=300)

fig.suptitle("Breast Quadrant vs bresat cancer")

#################################### Benign #################################

wedges, texts, autotexts = ax[0].pie(Breast_Quadrant_count,
                                  autopct=lambda pct: func(pct, Breast_Quadrant_count),

                                  labels=Breast_Quadrant_uniq,
                                  shadow=True,
                                  explode =explode,
                                  colors=palette_color,
                                  startangle=90,
                                  wedgeprops=wp,
                                  textprops=dict(color="steelblue"))


plt.setp(autotexts, size=9, weight="bold")
ax[0].set_title("Breast Quadrant and Benign Tumor",pad=60)
ax[0].legend(title = "Breast Quadrant")

################################# malignant #################################

wedges, texts, autotexts = ax[1].pie(Breast_Quadrant_disease_count,
                                  autopct=lambda pct: func(pct, Breast_Quadrant_disease_count),

                                  labels=Breast_Quadrant_disease_uniq,
                                  shadow=True,
                                  explode =explode,
                                  colors=palette_color,
                                  startangle=90,
                                  wedgeprops=wp,
                                  textprops=dict(color="steelblue"))


plt.setp(autotexts, size=9, weight="bold")
ax[1].set_title("Breast Quadrant and Malignant Tumor",pad=60)
ax[1].legend(title = "Breast Quadrant",loc ="upper left")

plt.tight_layout()
plt.savefig('Breast_Cancer_Detection/Results/Breast_Quadrant_piechart.png')

#Metastatis Pie Chart

palette_color = sns.color_palette("pastel")

# Wedge properties
wp = {'linewidth': 1, 'edgecolor': "green"}

# Creating autocpt arguments
explode = (0.1, 0.3)

def func(pct, allvalues):
    absolute = int(pct / 100.*np.sum(allvalues))
    return "{:.1f}%".format(pct, absolute)

fig,ax = plt.subplots(nrows=1, ncols =2 , figsize=(7,7),dpi=100)

fig.suptitle("Metastasis vs bresat cancer")

#################################### Benign #################################

wedges, texts, autotexts = ax[0].pie(Metastasis_count,
                                  autopct=lambda pct: func(pct, Metastasis_count),

                                  labels=Metastasis_uniq,
                                  shadow=True,
                                  explode =explode,
                                  colors=palette_color,
                                  startangle=90,
                                  wedgeprops=wp,
                                  textprops=dict(color="steelblue"))


plt.setp(autotexts, size=9, weight="bold")
ax[0].set_title("Metastasis and Benign Tumor",pad=60)
ax[0].legend(title=("Metastasis"))

################################# malignant #################################

wedges, texts, autotexts = ax[1].pie(Metastasis_disease_count,
                                  autopct=lambda pct: func(pct, Metastasis_disease_count),

                                  labels=Metastasis_disease_uniq,
                                  shadow=True,
                                  explode =explode,
                                  colors=palette_color,
                                  startangle=90,
                                  wedgeprops=wp,
                                  textprops=dict(color="steelblue"))


plt.setp(autotexts, size=9, weight="bold")
ax[1].set_title("Metastasis and Malignant Tumor",pad=60)
ax[1].legend(title=("Metastasis"))

plt.tight_layout()
plt.savefig('Breast_Cancer_Detection/Results/Metastasis_piechart.png')

#InvNodes Pie Chart
palette_color = sns.color_palette("pastel")

# Wedge properties
wp = {'linewidth': 1, 'edgecolor': "green"}

# Creating autocpt arguments
explode = (0.1, 0.3)

def func(pct, allvalues):
    absolute = int(pct / 100.*np.sum(allvalues))
    return "{:.1f}%".format(pct, absolute)

fig,ax = plt.subplots(nrows=1, ncols =2 , figsize=(7,7),dpi=100)

fig.suptitle("Inv_Nodes vs bresat cancer")

#################################### Benign #################################

wedges, texts, autotexts = ax[0].pie(Inv_Nodes_count,
                                  autopct=lambda pct: func(pct, Inv_Nodes_count),

                                  labels=Inv_Nodes_uniq,
                                  shadow=True,
                                  explode =explode,
                                  colors=palette_color,
                                  startangle=90,
                                  wedgeprops=wp,
                                  textprops=dict(color="steelblue"))


plt.setp(autotexts, size=9, weight="bold")
ax[0].set_title("Inv_Nodes and Benign Tumor",pad=60)
ax[0].legend(title=("Inv_Nodes"))

################################# malignant #################################

wedges, texts, autotexts = ax[1].pie(Inv_Nodes_disease_count,
                                  autopct=lambda pct: func(pct, Inv_Nodes_disease_count),

                                  labels=Inv_Nodes_disease_uniq,
                                  shadow=True,
                                  explode =explode,
                                  colors=palette_color,
                                  startangle=90,
                                  wedgeprops=wp,
                                  textprops=dict(color="steelblue"))


plt.setp(autotexts, size=9, weight="bold")
ax[1].set_title("Inv_Nodes and Malignant Tumor",pad=60)
ax[1].legend(title=("Inv_Nodes"))

plt.tight_layout()
plt.savefig('Breast_Cancer_Detection/Results/Inv_Nodes_piechart.png')

# History pie chart
palette_color = sns.color_palette("pastel")

# Wedge properties
wp = {'linewidth': 1, 'edgecolor': "green"}

# Creating autocpt arguments
explode = (0.1, 0.1)

def func(pct, allvalues):
    absolute = int(pct / 100.*np.sum(allvalues))
    return "{:.1f}%".format(pct, absolute)

fig,ax = plt.subplots(nrows=1, ncols =2 , figsize=(7,7),dpi=100)

fig.suptitle("Family history of cancer vs bresat cancer")

#################################### Benign #################################

wedges, texts, autotexts = ax[0].pie(History_count,
                                  autopct=lambda pct: func(pct, History_count),

                                  labels=History_uniq,
                                  shadow=True,
                                  explode =explode,
                                  colors=palette_color,
                                  startangle=90,
                                  wedgeprops=wp,
                                  textprops=dict(color="steelblue"))


plt.setp(autotexts, size=9, weight="bold")
ax[0].set_title("history and Benign Tumor",pad=60)
ax[0].legend(title=("History"),loc="upper left")

################################# malignant #################################

wedges, texts, autotexts = ax[1].pie(History_disease_count,
                                  autopct=lambda pct: func(pct, History_disease_count),

                                  labels=History_disease_uniq,
                                  shadow=True,
                                  explode =explode,
                                  colors=palette_color,
                                  startangle=90,
                                  wedgeprops=wp,
                                  textprops=dict(color="steelblue"))


plt.setp(autotexts, size=9, weight="bold")
ax[1].set_title("history and Malignant Tumor",pad=60)
ax[1].legend(title=("History"))
plt.tight_layout()
plt.savefig('Breast_Cancer_Detection/Results/History_piechart.png')
plt.close()

plt.figure(figsize=(10, 10))
sns.heatmap(dataset.corr(), annot=True, square=True, fmt='.2f')
plt.title('The correlation among features')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('Breast_Cancer_Detection/Results/correlation_matrix_heatplot.png')




