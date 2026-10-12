import control as ct 
import matplotlib.pyplot as plt 
# G(s) = 1/ (s^3 + 3s^2 + s + 2)
sys = ct.tf([1], [1,3,2,0])
ct.root_locus(sys)
plt.show()