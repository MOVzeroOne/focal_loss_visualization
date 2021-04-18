import matplotlib.pyplot as plt  
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np




def focal_loss_pos(x,alpha,gamma):
    """
    Focal Loss for Dense Object Detection https://arxiv.org/abs/1708.02002
    """
    pos_loss = -alpha * (1 - x)**gamma * np.log(x + 0.0000000000001)
    return pos_loss


def focal_loss_neg(x,alpha,gamma):
    """
    Focal Loss for Dense Object Detection https://arxiv.org/abs/1708.02002
    """
    neg_loss = -(1-alpha) * x**gamma * np.log(1-x + 0.0000000000001)
    return neg_loss




def focal_loss_pos_dy_dx(x,alpha,gamma):
    dx = x[1] - x[0] 
    y = focal_loss_pos(x,alpha,gamma)
    return np.gradient(y, dx)

def focal_loss_neg_dy_dx(x,alpha,gamma):
    dx = x[1] - x[0] 
    y = focal_loss_neg(x,alpha,gamma)
    return np.gradient(y, dx)



class focal_loss_visualizer():
    def __init__(self):
        self.x = np.linspace(0, 1, 1000)

        fig,self.axis = plt.subplots()
  

        plt.subplots_adjust(bottom=0.25)
        axSlider_1 = plt.axes([0.1,0.15,0.8,0.05])
        axSlider_2 = plt.axes([0.1,0.05,0.8,0.05])

        slider_alpha = Slider(axSlider_1,"alpha",0,1,0.25)
        slider_gamma = Slider(axSlider_2,"gamma",0,10,2)
        self.alpha = slider_alpha.val 
        self.gamma = slider_gamma.val
        self.update_graph()
        slider_alpha.on_changed(self.update_alpha)
        slider_gamma.on_changed(self.update_gamma)
        plt.show()

    def title(self):
        self.axis.set_title("focal loss")
    
    def update_alpha(self,value):
        self.alpha = value
        self.update_graph()

    def update_gamma(self,value):
        self.gamma = value 
        self.update_graph()

    def update_graph(self):
        self.axis.cla()
        self.axis.plot(self.x,focal_loss_pos(self.x,self.alpha,self.gamma),color="green")
        self.axis.plot(self.x,focal_loss_neg(self.x,self.alpha,self.gamma),color="red")
        self.title()

class focal_loss_visualizer_dy_dx():
    def __init__(self):
        self.x = np.linspace(0, 1, 1000)
         
        self.zoom = 1

        fig,self.axis = plt.subplots()
    

        plt.subplots_adjust(bottom=0.25)
        axSlider_1 = plt.axes([0.1,0.15,0.8,0.05])
        axSlider_2 = plt.axes([0.1,0.09,0.8,0.05])
        axSlider_3 = plt.axes([0.1,0.03,0.8,0.05])
        slider_alpha = Slider(axSlider_1,"alpha",0,1,0.25)
        slider_gamma = Slider(axSlider_2,"gamma",0,10,2)
        slider_zoom = Slider(axSlider_3,"zoom",0,1,0.5)
        
        self.alpha = slider_alpha.val 
        self.gamma = slider_gamma.val
        self.update_zoom(slider_zoom.val)
        self.update_graph()

        slider_alpha.on_changed(self.update_alpha)
        slider_gamma.on_changed(self.update_gamma)
        slider_zoom.on_changed(self.update_zoom)
        plt.show()

    def title(self):
        self.axis.set_title("focal loss dy/dx")
    

    def update_graph(self):
        self.axis.cla()
        self.axis.plot(self.x,focal_loss_pos_dy_dx(self.x,self.alpha,self.gamma),color="green")
        self.axis.plot(self.x,focal_loss_neg_dy_dx(self.x,self.alpha,self.gamma),color="red")
        self.title()

    def update_alpha(self,value):
        self.alpha = value
        self.update_graph()

    def update_gamma(self,value):
        self.gamma = value 
        self.update_graph()

    def update_zoom(self,value):
        self.zoom_on_x(value) 
        self.update_graph()

    def zoom_on_x(self,zoom):
        self.zoom = zoom
        middel_point = 0.5 
        max_value = 1 
        self.x = np.linspace(middel_point*self.zoom, max_value-middel_point*self.zoom, 1000)



if __name__ == "__main__":
    focal_loss_visualizer()
    focal_loss_visualizer_dy_dx()
    plt.show()


