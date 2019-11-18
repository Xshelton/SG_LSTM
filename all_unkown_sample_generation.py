import pandas as pd
global mirna_label
global gene_label

df=pd.read_csv('SG128_383_miRNA.csv')
mirna_label=df['label']
df=df.add_suffix('_m')

df2=pd.read_csv('SG128_317_gene.csv')
gene_label=df2['label']
df2=df2.add_suffix('_g')


df3=pd.read_csv('AXB_383_gene_default',encoding='ANSI')#
test_mirna=df3['0_mirna']
test_gene=df3['1_gene']
print('mirna',len(mirna_label))
print('gene',len(gene_label))
list1=[]
for i in range(0,len(df3)):
    list1.append({'mirna':test_mirna[i],'gene':test_gene[i]})
print(len(list1))

count=0
count_global=0
pre=0
print(mirna_label[365])
print(gene_label[279])
#fm.to_csv('unkown samples.csv',index=None)
#365 hsa-miR-548ap-3p 281 IP6K1
mm=len(df)
not_no=0
for i in range(0,mm):
    print(i)
    #if i==365:
    #    con=281;
    #else:
    con=0;
    for j in range(con,len(gene_label)):
        if j==len(gene_label)-1:#
            print(i)
            print(count_global,'/',len(mirna_label)*len(gene_label)-not_no)
        #设立flag
        flag={'mirna':mirna_label[i],'gene':gene_label[j]}#
        if flag in list1:
            not_no+=1
        else:
          mirna_i=df[i:i+1]
     
          gene_j=df2[j:j+1]
          mirna_i=mirna_i.reset_index(drop=True)
          gene_j=gene_j.reset_index(drop=True)
          temp=pd.concat([mirna_i,gene_j],axis=1)#1是加到后面
          step=10000
          if count==0:
            feature=temp
            count+=1
            count_global+=1
          else:
            feature=pd.concat([feature,temp],axis=0)#0是加到下面
            count+=1
            count_global+=1
        
          if count==step:
            feature=feature.reset_index(drop=True)
            label_m=feature['label_m']
            feature=feature.drop(['label_m'],axis=1)
            feature['label_m']=label_m
            feature.to_csv('unkown_samples{}to{}.csv'.format(pre,pre+step),index=None)
            pre=count_global
            print('save suc')
            count=0;
          if i==len(mirna_label)-1 and j==len(gene_label)-1:
            print('last one')
            feature=feature.reset_index(drop=True)
            label_m=feature['label_m']
            feature=feature.drop(['label_m'],axis=1)
            feature['label_m']=label_m 
            feature.to_csv('unkown_samples{}to{}.csv'.format(pre,count_global),index=None)
            print('save suc')

            

