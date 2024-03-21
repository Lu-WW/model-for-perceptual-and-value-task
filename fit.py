import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
class Model():
  def __init__(self) -> None:


    self.x0=[0,0,0.7,3,1,0.01,500]
    self.bounds=[(0,300),(0,300),(0,1),(0,10),(0.01,10),(0.,0.03),(0,5000)]
    self.maxt=5000

    self.dt=10
    self.conf0=0
    self.t0=0
    self.base_1=0
    self.base_2=0
    self.b0=300
    self.delta=0.01
    self.v0=0.1
    self.lamb=0
    self.tau=1e9

    self.lapse=1e-2

    self.w_unchosen=1


  def get_sub_data(self,ori_data,sub):
    # value task data
    sti_1=ori_data['sti_left_p'][:,0]*ori_data['sti_left_v'][:,0]
    sti_2=ori_data['sti_right_p'][:,0]*ori_data['sti_right_v'][:,0]
    choice=ori_data['choice'][:,0]
    rt=ori_data['rtime'][:,0]
    conf=ori_data['origconf'][:,0]
    

    data=np.zeros((choice[sub].shape[0],5))
    data[:,0]=choice[sub][:,0]+1
    data[:,1]=rt[sub][:,0]*1000
    data[:,2]=conf[sub][:,0]
    data[:,3]=sti_1[sub][:,0]/100
    data[:,4]=sti_2[sub][:,0]/100

    return data
  

  # def get_sub_data(self,ori_data,sub):
  #   # perceptual task data
  #   coh=ori_data['coh'][:,0]
  #   choice=ori_data['answer'][:,0]
  #   rt=ori_data['rtime'][:,0]
  #   conf=ori_data['conf'][:,0]

  #   data=np.zeros((coh[sub].shape[0],5))
  #   data[:,0]=choice[sub][:,0]
  #   data[:,1]=rt[sub][:,0]*1000
  #   data[:,2]=conf[sub][:,0]/16-1/32+0.5
  #   data[:,3]=coh[sub][:,0]/100
  #   data[:,4]=coh[sub][:,1]/100

  #   return data
  
  
  def init_paras(self,paras):
    
    
    self.base_1,self.base_2,self.conf0,self.k,self.sigma,self.lamb,self.tau=paras


  def simulate(self,v1,v2):
    
    x=np.zeros((int(self.maxt/self.dt),2))
    x[0,0]+=self.base_1
    x[0,1]+=self.base_2
    A=np.array([[1-self.delta*self.dt,-self.lamb*self.dt],[-self.lamb*self.dt,1-self.delta*self.dt]])
    b=np.array([self.k*v1*self.dt+self.v0*self.dt,self.k*v2*self.dt+self.v0*self.dt])
    dsigma=self.sigma*np.sqrt(self.dt)
    choice=-100
    rt=-100
    conf=-100
    for t in range(1,x.shape[0]):
      x[t]=x[t-1]@A+b+dsigma*np.random.randn(2)
      x[t][x[t]<0]=0
    
      if x[t][0]>self.get_boundary(t*self.dt) and x[t][0]>x[t][1]:
        choice=1
        rt=t*self.dt+self.t0
        conf=self.evd2conf(x[t][0]-self.w_unchosen*x[t][1])
        break
        
      if x[t][1]>self.get_boundary(t*self.dt):
        choice=2
        rt=t*self.dt+self.t0
        conf=self.evd2conf(x[t][1]-self.w_unchosen*x[t][0])
        break


    # if rt==-1:
    #   print(v1,v2,x)
    return choice,rt,conf
  
  
  def evd2conf(self,x):
    ret=(np.array(x)/self.b0)
    return ret
  

  def get_boundary(self,t):
    return self.b0*(1-0.9*t/(t+self.tau))




  def get_trial_score(self,data,repn=10,prob=None):
    
    choice,rt,conf,v1,v2=data
   
    if prob is None:
      model_data=np.zeros((repn,3))
      for i in range(repn):
        pd_choice,pd_rt,pd_conf=self.simulate(v1,v2)
        pd_conf=int(pd_conf>self.conf0)
        model_data[i,:]=np.array((pd_choice,pd_rt,pd_conf))
        
      p=(np.sum((model_data[:,0]==choice)*(model_data[:,2]==conf))/model_data.shape[0])
      score=-np.log(p*(1-self.lapse*4)+self.lapse)
    else:
      score=-np.log(prob[int(choice==2),int(conf==1)])
    return score
  
    
  def get_sub_score(self,paras,data,max_trial=999,mode=1,repn=10,prob=None):
    self.init_paras(paras)
    score=0

    import copy
    to_fit=copy.deepcopy(data)
    to_fit[:,2]=to_fit[:,2]>0.75
    if mode==1:
      loop=range(0,np.minimum(max_trial,to_fit.shape[0]),2)
    elif mode==2:
      loop=range(1,np.minimum(max_trial,to_fit.shape[0]),2)
    else:
      np.random.shuffle(to_fit)
      loop=range(np.minimum(max_trial,to_fit.shape[0]))
    for i in loop:
      if prob is None:
        score+=self.get_trial_score(to_fit[i],repn=repn)
      else:
        score+=self.get_trial_score(to_fit[i],repn=repn,prob=prob[i])
      
    return score
  


  def get_trial_ll(self,data,repn=10):
    
    choice,rt,conf,v1,v2=data
   
    model_data=np.zeros((repn,3))
    for i in range(repn):
      pd_choice,pd_rt,pd_conf=self.simulate(v1,v2)
      pd_conf=int(pd_conf>self.conf0)
      model_data[i,:]=np.array((pd_choice,pd_rt,pd_conf))
      
    p=(np.sum((model_data[:,0]==choice)*(model_data[:,2]==conf))/model_data.shape[0])
    score=-np.log(p*(1-self.lapse*4)+self.lapse)

    rt=model_data[:,1]
    return score,rt
  
  def get_sub_ll(self,paras,data,max_trial=999,mode=1,repn=10):
    self.init_paras(paras)
    score=0

    import copy
    to_fit=copy.deepcopy(data)
    to_fit[:,2]=to_fit[:,2]>0.75
    if mode==1:
      loop=range(0,np.minimum(max_trial,to_fit.shape[0]),2)
    elif mode==2:
      loop=range(1,np.minimum(max_trial,to_fit.shape[0]),2)
    else:
      np.random.shuffle(to_fit)
      loop=range(np.minimum(max_trial,to_fit.shape[0]))
    rt_list=[]
    data_rt=[]
    dv_list=[]
    for i in loop:
      s,rt=self.get_trial_ll(to_fit[i],repn=repn)
      score+=s
      rt_list.append(rt)
      data_rt.append(data[i,1])
      dv_list.append(data[i,-2]-data[i,-1])
    rt_list=np.array(rt_list)
    dv_list=np.array(dv_list)
    dv=data[:,-2]-data[:,-1]
    rt=self.transform(rt_list,data[:,1],np.broadcast_to(dv_list.reshape((-1,1)),rt_list.shape),dv)
    rt=np.mean(rt,-1)
    ssr=np.sum((rt-np.array(data_rt))**2)
    n=rt.shape[0]
    score+=0.5*np.log(ssr/n)*n
      
    return score
  
  def get_prob(self,data,repn=100):
    prob=np.zeros((data.shape[0],2,2))
    for i in range(data.shape[0]):
      choice,rt,conf,v1,v2=data[i]
      model_data=np.zeros((repn,3))
      for r in range(repn):
        pd_choice,pd_rt,pd_conf=self.simulate(v1,v2)
        pd_conf=int(pd_conf>self.conf0)
        model_data[r,:]=np.array((pd_choice,pd_rt,pd_conf))

      for x in range(2):
        for y in range(2):
          prob[x,y]=(np.sum((model_data[:,0]==x+1)*(model_data[:,2]==y))/model_data.shape[0])

      prob=prob*(1-self.lapse*4)+self.lapse
    return prob



  def transform(self,source,target,v1=None,v2=None):

    if v1 is None or v2 is None:
      import copy
      ret=copy.deepcopy(source)
      
      ret-=np.mean(ret)
      ret/=np.std(ret)
      ret*=np.std(target)
      ret+=np.mean(target)
      return ret
    from scipy.stats import linregress
    nq=10

    e=np.quantile(v2,np.linspace(0,1,nq+1)) 
    m1= (np.histogram(v1, e, weights=source)[0] /np.histogram(v1, e)[0])
    m2= (np.histogram(v2, e, weights=target)[0] /np.histogram(v2, e)[0])
    res=linregress(m1,m2)

    ret=source*res.slope+res.intercept
    return ret
  
  def predict(self,paras,data,repn=10):
    self.init_paras(paras)
    prd=np.zeros((data.shape[0],repn,3))
    ret=np.zeros((data.shape[0],3))


      
    def get_prd(data,repn=10):
      choice,rt,conf,v1,v2=data
      
      
      ret=np.zeros((repn,3))
      for i in range(repn):
        pd_choice=-1
        while (pd_choice<0):
          pd_choice,pd_rt,pd_conf=self.simulate(v1,v2)
        # pd_conf=int(pd_conf>self.conf0)
        ret[i]=np.array([pd_choice,pd_rt,pd_conf])
        
      return ret
    for i in range(data.shape[0]):
      
      prd[i]=get_prd(data[i],repn=repn)


    ret[:,0]=np.mean(prd[:,:,0],1)

    dv=(data[:,-2]-data[:,-1]).reshape((-1,1))
    dv=np.broadcast_to(dv,prd[:,:,1].shape)
    prd[:,:,1]=self.transform(prd[:,:,1],data[:,1],dv,dv[:,0])
    ret[:,1]=np.mean(prd[:,:,1],1)
    m=0.75
    
    idx=prd[:,:,2]>self.conf0
    prd[:,:,2][idx]=self.transform(prd[:,:,2][idx],data[:,2][data[:,2]>m])
    prd[:,:,2][~idx]=self.transform(prd[:,:,2][~idx],data[:,2][data[:,2]<=m])


    ret[:,2]=np.mean(prd[:,:,2],1)

    return ret



class No_mi_model(Model):
  def __init__(self) -> None:
    super().__init__()

    self.x0=[0,0,0,3,1,1000]
    self.bounds=[(0,300),(0,300),(-1,1),(0,30),(0.01,20),(0,5000)]

  def init_paras(self,paras):
    self.base_1,self.base_2,self.conf0,self.k,self.sigma,self.tau=paras

class No_urg_model(Model):
  def __init__(self) -> None:
    super().__init__()
    self.x0=[0,0,0,3,1,0.001]
    self.bounds=[(0,300),(0,300),(-1,1),(0,30),(0.01,20),(0,0.03)]

  def init_paras(self,paras):
    self.base_1,self.base_2,self.conf0,self.k,self.sigma,self.lamb=paras

class Minimal_model(Model):
  def __init__(self) -> None:
    super().__init__()
    self.x0=[0,0,0,3,1]
    self.bounds=[(0,300),(0,300),(-1,1),(0,30),(0.01,20)]

  def init_paras(self,paras):
    self.base_1,self.base_2,self.conf0,self.k,self.sigma=paras



class Model_2stage(Model):
  def __init__(self) -> None:
    super().__init__()
    self.x0=[0,0,0.7,3,1,0.01,500,1]
    self.bounds=[(0,300),(0,300),(0,1),(0,10),(0.01,10),(0.,0.03),(0,5000),(0,2)]
    

  def init_paras(self,paras):
    self.base_1,self.base_2,self.conf0,self.k,self.sigma,self.lamb,self.tau,self.w_unchosen=paras

class No_mi_model_2stage(Model):
  def __init__(self) -> None:
    super().__init__()
    self.x0=[0,0,0.,3,1,500,1]
    self.bounds=[(0,300),(0,300),(-1,1),(0,30),(0.01,20),(0,5000),(0,2)]

  def init_paras(self,paras):
    self.base_1,self.base_2,self.conf0,self.k,self.sigma,self.tau,self.w_unchosen=paras

class No_urg_model_2stage(Model):
  def __init__(self) -> None:
    super().__init__()
    self.x0=[0,0,0.,3,1,0.01,1]
    self.bounds=[(0,300),(0,300),(-1,1),(0,30),(0.01,20),(0,0.03),(0,2)]

  def init_paras(self,paras):
    self.base_1,self.base_2,self.conf0,self.k,self.sigma,self.lamb,self.w_unchosen=paras

class Minimal_model_2stage(Model):
  def __init__(self) -> None:
    super().__init__()
    self.x0=[0,0,0.,3,1,1]
    self.bounds=[(0,300),(0,300),(-1,1),(0,30),(0.01,20),(0,2)]

  def init_paras(self,paras):
    self.base_1,self.base_2,self.conf0,self.k,self.sigma,self.w_unchosen=paras




def eval(model_class,ori_data):
  model=model_class()
  all_res=np.load(f'{model_class.__name__}_paras.npy',allow_pickle=True)
  dat=np.array([r.x for r in all_res])
  
  
  res=np.zeros((ori_data.shape[0],3))
  all_data=np.zeros((0,5))
  all_prd=np.zeros((0,3))
  prob_list=[]
  prd_list=[]
  for sub in range(ori_data.shape[0]) :
    data=model.get_sub_data(ori_data,sub)


    all_data=np.concatenate((all_data,data),0)
    paras=dat[sub]
    model.init_paras(paras)
    
    
    print(f'Sub id:{sub}')
    print(f'Paras:{paras}')

    prob=model.get_prob(data,repn=100)
    prob_list.append(prob)
    res[sub,0]=model.get_sub_ll(paras,data,9999,2,repn=100)
    k=len(paras)
    res[sub,1]=2*k+2*res[sub,0]
    res[sub,2]=np.log(data.shape[0])*k+2*res[sub,0]
    print(f'Score:{res[sub,0]}')
    print(f'AIC:{res[sub,1]}')
    print(f'BIC:{res[sub,2]}')
    
    prd=model.predict(paras,data,repn=100)
    prd_list.append(prd)
    all_prd=np.concatenate((all_prd,prd),0)
 
    pass
  # np.save(f'{model_class.__name__}_prd.npy',prd_list)
  # np.save(f'{model_class.__name__}_prob.npy',prob_list)
  # np.save(f'{model_class.__name__}_res.npy',res)
  np.save(f'{model_class.__name__}_eval.npy',{'probability':prob_list,'prediction':prd_list,'score':res})
  data=np.array(all_data)
  prd=np.array(all_prd)
  # np.save(f'{model_class.__name__}_data.npy',data)
  dv=np.around(data[:,3]-data[:,4],6)
  dvs=np.unique(dv)
  sv=np.around(data[:,3]+data[:,4],6)
  svs=np.unique(sv)
  
  l=['Choice','Reaction time (ms)','Confidence']
  plt.figure(figsize=(12,4))
  for p in range(3):
    ave_sub=np.zeros_like(dvs)
    ave_mod=np.zeros_like(dvs)
    for i in range((dvs.shape[0])):
      idx=dv==dvs[i]
      ave_sub[i]=np.mean(data[idx,p])
      ave_mod[i]=np.mean(prd[idx,p])
      
    if p==0:
      ave_sub=2-ave_sub
      ave_mod=2-ave_mod
    if p==2:
      ave_sub=ave_sub/2+0.5
      ave_mod=ave_mod/2+0.5
    # plt.figure()
    plt.subplot(1,3,p+1)
    plt.plot(dvs,ave_sub,label='data')
    plt.plot(dvs,ave_mod,label='model')
    plt.xlabel('Coherence difference')
    plt.ylabel(l[p])
    plt.gca().spines[['top','right']].set_visible(False)
  # plt.show(block=False)
  plt.savefig(f'eval_{model_class.__name__}')
  return res


def fit(model_class,ori_data):
  
  model=model_class()
  

  to_save=[]
  for sub in range(ori_data.shape[0]) :
    data=model.get_sub_data(ori_data,sub)
    
    
    from scipy.optimize import differential_evolution
    res=differential_evolution(model.get_sub_score,model.bounds,args=(data,),polish=False,maxiter=50,x0=model.x0,workers=16,disp=True)
    
    print(res)
    to_save.append(res)
    np.save(f'{model_class.__name__}_paras.npy',to_save)

    pass
if __name__=='__main__':
 
  
  from scipy.io import loadmat
  ori_data=loadmat('./value_data')['data_final']
  # ori_data=loadmat('./data')['origdata']

  model_list=[Model,Minimal_model,No_mi_model,No_urg_model,
            Model_2stage,Minimal_model_2stage,No_mi_model_2stage,No_urg_model_2stage,]

  for m in model_list:
    fit(m,ori_data)
    eval(m,ori_data)
    
  
  