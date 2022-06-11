import scipy.io
mat=scipy.io.loadmat('meta.mat')

for i,s in enumerate(mat['synsets']):
    if s[0][1]=='n15075141':
       print(i,s[0])
       print('-----------------')
   
