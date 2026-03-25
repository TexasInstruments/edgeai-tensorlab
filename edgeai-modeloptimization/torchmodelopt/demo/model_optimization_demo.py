import streamlit as st


# import matplotlib.pyplot as plt
from plotly import graph_objects as go, express as px
import pandas as pd

import datetime

import back_end


st.markdown("## Edgeai Model Optimazation Kit Demo")
# st.info("Use the side add the details")


def human_format(num):
    num = float('{:.3f}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:.2f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])    


def _plot(x_key,y_key):
    df = pd.read_csv('acc_flop_param.csv')
    # print(df[x_key],'\n',df[y_key])
    fig = px.scatter(df,x=x_key,y=y_key,title=f'{str(x_key).capitalize()} vs {str(y_key).capitalize()}',hover_name='user',hover_data=[x_key,y_key,'date','time','surgery_rules','sparsity_ratio','sparsity_type','quantization'])
    for i, l in enumerate(df['user']):
        fig.add_annotation(x=df[x_key][i],y=df[y_key][i],text=l,)
    # fig.update_xaxes(labelalias= {x:human_format(int(x)) for x in fig.select_xaxes()})        
    return fig


if 'flop_graph' not in list(st.session_state.keys()):
    st.session_state.flop_graph = _plot('flops','accuracy')
if 'param_graph' not in list(st.session_state.keys()):
    st.session_state.param_graph = _plot('params','accuracy')

st.write(st.session_state.flop_graph)
st.write(st.session_state.param_graph)


st.sidebar.image("icon.png")


if ("user_name" in list(st.session_state.keys())) and  len(st.session_state.user_name):
    new_name = st.sidebar.text_input(f"Enter Your Name:",value=st.session_state.user_name)
    if new_name!=st.session_state.user_name:
        ok_btn = st.sidebar.button("OK",) 
        if ok_btn:
            if len(new_name):
                st.session_state.update(user_name=new_name)
                st.sidebar.success(f"Successfully changed User Name to {st.session_state.user_name}")
            else: 
                st.sidebar.error('No User Name given.','Give a User Name')    
else:
    user_name = st.sidebar.text_input("Enter Your Name:",)    
    ok_btn = st.sidebar.button("OK",key=1) 
    if ok_btn:
        if len(user_name):
            st.session_state.update(user_name=user_name)
            st.sidebar.success(f"Successfully saved User Name as {st.session_state.user_name}")
        else: 
            st.sidebar.error('No User Name given.','Give a User Name')


st.sidebar.header("Training Details")
# st.sidebar.write('model =)
model_type = 'mobilenet_v2'
st.sidebar.write('Model: **Mobilenet V2**')
# model_type = st.sidebar.selectbox('Select a model',['mobilenet_v2', 'efficientnet_b0', 'convnext_tiny', 'regnet_y_8gf'])
st.sidebar.markdown("Dataset: [**Imagenette**](https://github.com/fastai/imagenette/tree/master)")


st.markdown("### Surgery:")
col1,col2 = st.columns(2)
if not ("surgery_dict" in list(st.session_state.keys())):
    st.session_state.update(surgery_dict = dict())


with col1:
    st.markdown("#### From")
with col2:
    st.markdown("#### To")


st.sidebar.header('Surgery Rules:')
key = st.sidebar.selectbox("From:",['<Select One>','GELU','ReLU6','ReLU'])
val = st.sidebar.selectbox("To:",['<Select One>','GELU','ReLU'])


add_btn = st.sidebar.button('ADD',)
if add_btn:
    if key != '<Select One>' and val != '<Select One>':
        st.session_state.surgery_dict.update({key:val})
        st.sidebar.success('Successfully Added')
    else:    
        st.sidebar.error("Choose a Correct Coneversion")


if "surgery_dict" in list(st.session_state.keys()):
    if len(st.session_state.surgery_dict) == 0:
        st.info("No Surgery Specified Now!")
    with col1:
        # st.write(key)
        for k in st.session_state.surgery_dict:
            st.write(k)
    with col2:
        # st.write(val)
        for v in st.session_state.surgery_dict.values():
            st.write(v)


if "numeric" not in list(st.session_state.keys()):
    st.session_state.numeric = 0.0


st.sidebar.header("Sparsity:")
if ("sp_ratio" in list(st.session_state.keys())):
    st.session_state.sp_ratio= st.sidebar.number_input("Enter Ratio between 0 and 1:",key= 'numeric',value=st.session_state.numeric,min_value=0.0,max_value=1.0,format='%.4f')
else:
    st.session_state.sp_ratio= st.sidebar.number_input("Enter Ratio between 0 and 1",key= 'numeric',value=st.session_state.numeric,min_value=0.0,max_value=1.0,format='%.4f')
    st.session_state.sp_ratio = st.session_state.numeric


if st.session_state.sp_ratio >0:
    sp_type = st.sidebar.radio('Type',['Unstructured','Channel'],horizontal=True) 
else:
    # st.sidebar.warning('With 0% sparsity ratio no sparsity type can be selected. ')
    sp_type = 'Channel'
    

st.markdown("### Sparsity:")
st.write(f"Ratio: **{st.session_state.sp_ratio}**")
st.write(f"Type: **{sp_type if st.session_state.sp_ratio != 0.0 else 'None'}**")


st.sidebar.header("Quantization:")
qntzn_option = ['Disabled','Enabled']
qntzn = st.sidebar.radio("# Enable Quanitization:",qntzn_option,horizontal=True,label_visibility='hidden') 


st.markdown("### Quanitization:")
st.write(f"**{qntzn}**")


if qntzn == 'Enabled' and st.session_state.sp_ratio > 0:
    st.warning("Both Sparsity and Quantization are not supported simultaneously.")


# n=2
# no_of_cols = 2
# cols = st.sidebar.columns(no_of_cols)


def rst_btn_task():
    st.session_state.surgery_dict.clear()
    st.session_state.user_name = ""
    st.session_state.numeric = 0.0
rst_btn = st.button("Reset",on_click=rst_btn_task)


train_btn = st.sidebar.button("Train")


def _write_to_csv(user,surgery_rules,sp_ratio,sp_type,quantization,acc,flops,params):
    dt = datetime.datetime.now()
    with open("acc_flop_param.csv",'a',) as file:
        file.write(f'\n{user},{dt:%d/%b/%Y},{dt:%I:%M:%S %p},{surgery_rules if len(surgery_rules) else "None"},{sp_ratio},{sp_type if sp_ratio >0 else "None"},{quantization},{acc},{flops}.0,{params}.0')


if train_btn:
    if "user_name" in list(st.session_state.keys())  and len(st.session_state.user_name):
        st.write("User Name:", st.session_state.user_name)
        st.write(f''' 
                Surgery:\n
                {st.session_state.surgery_dict}
                ''')
        st.write(f'''
                Sparsity:\n
                    Ratio: {st.session_state.sp_ratio}\n
                    Type: {sp_type if st.session_state.sp_ratio != 0.0 else 'None'}
                ''')
        st.write(f'''
                    Quantization: {qntzn}
                ''')
        if qntzn == 'Enabled' and st.session_state.sp_ratio > 0:
            st.error("Both Sparsity and Quantization are not supported simultaneously.Please disable atleast one!")
        else:
            if qntzn == 'Enabled':
                st.warning('Currently Quantization is not supported!\n Disabling Quantization for this run!')
                qntzn = 'Disabled'
            acc,flops,params= back_end.backend_task(model_type,surgery_dict=st.session_state.surgery_dict,sp_type=sp_type.lower(),sp_ratio= st.session_state.sp_ratio,qntzn=(qntzn=='Enabled'))
            st.success("Training Successful")
            st.balloons()
            st.success("We got accuracy {0:.2f}% with flop count as {1} and number of parameters as {2} .".format(acc,human_format(flops),human_format(params)))
            _write_to_csv(st.session_state.user_name,surgery_rules=st.session_state.surgery_dict,sp_type=sp_type,sp_ratio= st.session_state.sp_ratio,quantization=qntzn,acc=acc,flops=flops,params=params)
            st.session_state.flot_graph = _plot('flops','accuracy')
            st.session_state.param_graph = _plot('params','accuracy')
            st.rerun()
    else:
        st.error("Please Enter a username first")



