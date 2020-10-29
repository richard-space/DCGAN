import tensorflow as tf
#import matplotlib.pyplot as plt
#import numpy as np
import warnings
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
import numpy as np
import pathos.multiprocessing as mp

def maindo(images,noise_img,testImg):
    warnings.filterwarnings("ignore")
    #matplotlib inline
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph('./MyModel.meta')
    sess=tf.Session(config=tf.ConfigProto(log_device_placement=True))
    saver.restore(sess,'./MyModel')
    graph = tf.get_default_graph()
    inputs_noise=tf.get_collection('inputs_noise')[0]
    outputs=tf.get_collection('outputs')[0]
    
    
      
    def plot_images(samples):
    #    samples = (samples + 1) / 2
        img=samples.reshape((32, 32, 3))
    #    img=dwt_r(img)
    #    for x in range(0,32):
    #        for y in range(0,32):
    #            for z in range(0,3):
    #                if img[x,y,z]<0:
    #                    img[x,y,z]=0
        return img
    
    
    def clickReadData():
        nonlocal a , b, canvas
        test=noise_img[int(E0text.get())]
        true=images[int(E0text.get())]
        Img=test       
        examples_noise=Img.reshape(-1,32,32,3)
        result = sess.run(outputs, feed_dict = {inputs_noise : examples_noise})
        a.imshow(test)
        b.imshow(plot_images(result))
        print(np.sum(np.abs(true-result)))
        canvas.draw()
    
    #examples_noise=Img.reshape(-1,32*32*3)
    
    #plt.imshow(test)
    #plt.show()
    #plot_images(result)
    #plt.show()
    
    root =tk.Tk()
    root.title("matplotlib in TK")
    f =Figure(figsize=(5,4), dpi=100)
    a = f.add_subplot(121)
    b = f.add_subplot(122)
    
    
    canvas =FigureCanvasTkAgg(f, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    #把matplotlib绘制图形的导航工具栏显示到tkinter窗口上
    toolbar =NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    #定义并绑定键盘事件处理函数
    def on_key_event(event):
        print('you pressed %s'% event.key)
        key_press_handler(event, canvas, toolbar)
    canvas.mpl_connect('key_press_event', on_key_event)
    #按钮单击事件处理函数
    def _quit():
    #结束事件主循环，并销毁应用程序窗口
        sess.close()
        root.quit()
        root.destroy()
        
        
    button =tk.Button(master=root, text='Quit', command=_quit)
    button.pack(side=tk.BOTTOM)
    F1=tk.Frame(root)
    F1.pack()
    L0=tk.Label(F1,text='Input Data File')
    L0.pack(side='left')
    E0text=tk.StringVar()
    E0text.set('4520')
    E0=tk.Entry(F1,textvariable=E0text)
    E0.pack(side='left')
    B0=tk.Button(F1,text='Read Data',command=clickReadData)
    B0.pack(side='left')
    root.mainloop()

if __name__=='__main__':    
    mp.freeze_support()
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"


    pool = mp.Pool(1)
    temp=pool.apply_async(maindo,(images,noise_img,testImg,))         
#    print(temp.get())

    pool.close()# 将进程池关闭，不再接受新的进程
    pool.join()# 主进程阻
