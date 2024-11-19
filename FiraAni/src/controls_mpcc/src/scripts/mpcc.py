#! /usr/bin/python3

import numpy as np
from scipy.special import comb
import casadi
import sys
sys.path.append('/home/umic/ws/src/acados/interfaces/acados_template')
from acados_template.acados_ocp import AcadosOcp
from acados_template.acados_ocp_solver import AcadosOcpSolver
from acados_template.acados_model import AcadosModel
import rospy
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Float64
import matplotlib.pyplot as plt


class Contouring():

    def bernstein_poly(self,i,n,t):
        return comb(n,i)*(t**(n-i))*(1-t)**i

    def bezeier_curve(self,points, nTimes):
        nPoints = len(points[0])
        xPoints = points[0]
        yPoints = points[1]
        t = np.linspace(0.0, 1.0, nTimes)
        poly_arr = np.array([self.bernstein_poly(i, nPoints-1,t) for i in range(0, nPoints)])
        xvals = np.dot(xPoints, poly_arr)
        yvals = np.dot(yPoints, poly_arr)
        return np.array(xvals[::-1]), np.array(yvals[::-1])

    def bezier_derivative(self,points, nTimes):
        nPoints = len(points[0])
        xPoints = points[0]
        yPoints = points[1]
        dXPoints = np.array([(nPoints - 1) * (xPoints[i + 1] - xPoints[i]) for i in range(nPoints - 1)])
        dYPoints = np.array([(nPoints - 1) * (yPoints[i + 1] - yPoints[i]) for i in range(nPoints - 1)]) 
        t = np.linspace(0.0, 1.0, nTimes)
        poly_arr = np.array([self.bernstein_poly(i, nPoints - 2, t) for i in range(0, nPoints - 1)])
        dxvals = np.dot(dXPoints, poly_arr)
        dyvals = np.dot(dYPoints, poly_arr)
        return np.array(dxvals[::-1]), np.array(dyvals[::-1])

    def arc_length(self,x,slope,nTimes):   
        s = [0]
        length = 0
        for i in range(1,nTimes):
            length += np.sqrt(1 + slope[i]**2)*(x[i]-x[i-1])
            s.append(length)
        return s

    def generate_path_table(self,path,nTimes,r):
        table = []
        xbez,ybez = self.bezeier_curve(path,nTimes)
        dxbez,dybez = self.bezier_derivative(path,nTimes)
        phi = dybez/dxbez
        s = self.arc_length(xbez,phi,nTimes)
        d_o = r + xbez*(-np.sin(phi)) + ybez*(np.cos(phi))
        d_i = -r + xbez*(-np.sin(phi)) + ybez*(np.cos(phi))
        for i in range(nTimes):
            table.append([s[i],xbez[i],ybez[i],phi[i],np.cos(phi[i]),np.sin(phi[i]),d_o[i],d_i[i]])
        return np.array(table)


class Model():

    def kinematic_model(self,lwb):  

        model = casadi.types.SimpleNamespace()
        model_name = "kinematic_model"
        model.name = model_name
        xt =  casadi.SX.sym("xt")
        yt =  casadi.SX.sym("yt")
        phit = casadi.SX.sym("phit")
        sin_phit = casadi.SX.sym("sin_phit")
        cos_phit = casadi.SX.sym("cos_phit")
        theta_hat = casadi.SX.sym("theta_hat")
        Qc = casadi.SX.sym("Qc")
        Ql = casadi.SX.sym("Ql")
        Q_theta = casadi.SX.sym("Q_theta")
        R_d = casadi.SX.sym("R_d")
        R_delta = casadi.SX.sym("R_delta")
        r = casadi.SX.sym("r")
        p = casadi.vertcat(xt, yt, phit, sin_phit, cos_phit, theta_hat, Qc, Ql, Q_theta, R_d, R_delta, r)
        posx = casadi.SX.sym("posx")
        posy = casadi.SX.sym("posy")
        vx = casadi.SX.sym("vx")
        phi = casadi.SX.sym("phi")
        delta = casadi.SX.sym("delta")
        d = casadi.SX.sym("d")
        theta = casadi.SX.sym("theta")
        posxdot = casadi.SX.sym("xdot")
        posydot = casadi.SX.sym("ydot")
        vxdot = casadi.SX.sym("vxdot")
        phidot = casadi.SX.sym("phidot")
        deltadot = casadi.SX.sym("deltadot")
        thetadot = casadi.SX.sym("thetadot")
        ddot = casadi.SX.sym("ddot")

        u = casadi.vertcat(ddot, deltadot, thetadot)
        x = casadi.vertcat(posx, posy, phi, vx, theta, d, delta)
        xdot = casadi.vertcat(posxdot, posydot, phidot, vxdot, thetadot, ddot, deltadot)
        f_expl = casadi.vertcat(vx*casadi.cos(phi), vx*casadi.sin(phi), vx/lwb * casadi.tan(delta), d, thetadot, ddot, deltadot)
        z = casadi.vertcat([])

        model.f_expl_expr = f_expl
        model.f_impl_expr = xdot - f_expl
        model.x = x
        model.xdot = xdot
        model.u = u
        model.p = p
        model.z = z
    
        model.d_min = rospy.get_param('/controls_node/d_min')
        model.d_max = rospy.get_param('/controls_node/d_max')
        model.ddot_min = rospy.get_param('/controls_node/ddot_min')
        model.ddot_max = rospy.get_param('/controls_node/ddot_max')
        model.delta_min = np.deg2rad(rospy.get_param('/controls_node/delta_min'))
        model.delta_max = np.deg2rad(rospy.get_param('/controls_node/delta_max'))
        model.deltadot_min = rospy.get_param('/controls_node/deltadot_min')
        model.deltadot_max = rospy.get_param('/controls_node/deltadot_max')
        model.thetadot_min = rospy.get_param('/controls_node/thetadot_min')
        model.thetadot_max = rospy.get_param('/controls_node/thetadot_max')
        model.theta_min = rospy.get_param('/controls_node/theta_min')
        model.theta_max = rospy.get_param('/controls_node/theta_max')
        model.vx_min = rospy.get_param('/controls_node/vx_min')
        model.vx_max = rospy.get_param('/controls_node/vx_max')

        model.x0 = np.array([0, 0, 0, 1, 0, 0, 0])

        xt_hat = xt + cos_phit * ( theta - theta_hat)
        yt_hat = yt + sin_phit * ( theta - theta_hat)
        e_cont = sin_phit * (xt_hat - posx) - cos_phit *(yt_hat - posy)
        e_lag = cos_phit * (xt_hat - posx) + sin_phit *(yt_hat - posy)

        model.con_h_expr = e_cont**2+e_lag**2-(r)**2
        model.stage_cost = e_cont * Qc * e_cont + e_lag * Qc * e_lag - Q_theta * thetadot + ddot * R_d * ddot + deltadot * R_delta * deltadot
        return model

class Acados_cfg():

    def acados_settings_kin(self,Tf, N, lwb):

        ocp = AcadosOcp()

        model = Model().kinematic_model(lwb)

        model_ac = AcadosModel()
        model_ac.f_impl_expr = model.f_impl_expr
        model_ac.f_expl_expr = model.f_expl_expr
        model_ac.x = model.x
        model_ac.xdot = model.xdot
        model_ac.u = model.u
        model_ac.p = model.p
        model_ac.z = model.z
        model_ac.cost_expr_ext_cost = model.stage_cost
        model_ac.cost_y_expr = model.stage_cost
        model_ac.cost_y_expr_0 = model.stage_cost
        model_ac.con_h_expr = model.con_h_expr
        model_ac.name = model.name
        ocp.model = model_ac

        nx  = model.x.size()[0]
        nu  = model.u.size()[0]
        nz  = model.z.size()[0]
        np_  = model.p.size()[0]

        ocp.dims.nx   = nx
        ocp.dims.nz   = nz
        ocp.dims.nu   = nu
        ocp.dims.np   = np_
        ocp.dims.nh = 1
        ocp.dims.nbx = 4
        ocp.dims.nbu = nu
        ocp.dims.nsh = 1
        ocp.dims.ns = 1
        ocp.dims.N = N
        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.zu = 1000 * np.ones((ocp.dims.ns,))
        ocp.cost.zl = 1000 * np.ones((ocp.dims.ns,))
        ocp.cost.Zu = 1000 * np.ones((ocp.dims.ns,))
        ocp.cost.Zl = 1000 * np.ones((ocp.dims.ns,))

        ocp.constraints.uh = np.array([0.00])
        ocp.constraints.lh = np.array([-10])
        ocp.constraints.lsh = 0.1*np.ones(ocp.dims.nsh)
        ocp.constraints.ush = 0.001*np.ones(ocp.dims.nsh)
        ocp.constraints.idxsh = np.array([0])
        ocp.constraints.lbx = np.array([model.vx_min, model.theta_min, model.d_min, model.delta_min])
        ocp.constraints.ubx = np.array([model.vx_max, model.theta_max, model.d_max, model.delta_max])
        ocp.constraints.idxbx = np.array([3,4,5,6])
        ocp.constraints.lbu = np.array([model.ddot_min, model.deltadot_min, model.thetadot_min])
        ocp.constraints.ubu = np.array([model.ddot_max, model.deltadot_max, model.thetadot_max])
        ocp.constraints.idxbu = np.array([0, 1, 2])
        ocp.constraints.x0 = model.x0

        ocp.solver_options.tf = Tf
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.parameter_values = np.zeros(np_)

        ocp.solver_options.nlp_solver_step_length = 0.05
        ocp.solver_options.nlp_solver_max_iter = 100
        ocp.solver_options.tol = 1e-4

        acados_solver = AcadosOcpSolver(ocp)

        return acados_solver

class MPCC():

    def __init__(self):
        self.lwb = rospy.get_param('/controls_node/lwb')
        self.Tf = rospy.get_param('/controls_node/Tf')
        self.N = rospy.get_param('/controls_node/N')
        self.Qc = rospy.get_param('/controls_node/Qc')
        self.Ql = rospy.get_param('/controls_node/Ql')
        self.Q_theta = rospy.get_param('/controls_node/Q_theta')
        self.R_d = rospy.get_param('/controls_node/R_d')
        self.R_delta = rospy.get_param('/controls_node/R_delta')
        self.nTimes = rospy.get_param('/controls_node/nTimes')
        self.r = rospy.get_param('/controls_node/r')
        self.loop_rate = rospy.get_param('/controls_node/loop_rate')
        self.path = [[],[]]

        self.velocity_pub = rospy.Publisher('/velocity', Float64, queue_size=10)
        self.steer_pub = rospy.Publisher('/steer', Float64, queue_size=10)
    
        rospy.Subscriber('/best_trajectory', PoseArray, self.wp_cb)

    def wp_cb(self, data):
        self.x = []
        self.y = []
        self.path = [[],[]]
        for i in range(len(data.poses)):
            self.x.append(data.poses[i].position.x)
            self.y.append(data.poses[i].position.y)
        self.path = [self.x, self.y]

    def control_loop(self):

        rate = rospy.Rate(self.loop_rate)
        while not rospy.is_shutdown():
            if (len(self.path[0])==0):
                continue

            track_lu_table = Contouring().generate_path_table(self.path,self.nTimes,self.r)
            acados_solver = Acados_cfg().acados_settings_kin(self.Tf, self.N, self.lwb)

            vars = ['sval', 'xtrack', 'ytrack', 'phitrack', 'cos(phi)', 'sin(phi)', 'g_upper', 'g_lower']
            xt0 = track_lu_table[0,vars.index('xtrack')]
            yt0 = track_lu_table[0,vars.index('ytrack')]
            phit0 = track_lu_table[0,vars.index('phitrack')]
            theta_hat0 = track_lu_table[0,vars.index('sval')]
            x0 = np.array([xt0, yt0, 0, 0, theta_hat0, 0, 0])

            theta_old = theta_hat0*np.ones((self.N,))
            x_current = np.tile(x0,(self.N,1))
            theta_diff = 1e7

            while(theta_diff != 0):
                index_lin_points = 100*theta_old
                index_lin_points = index_lin_points.astype(np.int32)
                track_lin_points = track_lu_table[index_lin_points,:]

                for j in range(self.N):
                    p_val = np.array([track_lin_points[j,vars.index('xtrack')],
                                        track_lin_points[j,vars.index('ytrack')],
                                        track_lin_points[j,vars.index('phitrack')],
                                        track_lin_points[j,vars.index('sin(phi)')],
                                        track_lin_points[j,vars.index('cos(phi)')],
                                        track_lin_points[j,vars.index('sval')], 
                                        self.Qc, self.Ql, self.Q_theta, self.R_d, self.R_delta, self.r])
                    
                    x_val = x_current[j]
                    acados_solver.set(j,"p", p_val)
                    acados_solver.set(j,"x", x_val)

                acados_solver.set(0, "lbx", x0)
                acados_solver.set(0, "ubx", x0)
                acados_solver.solve()

                for k in range(self.N):
                    xsol = acados_solver.get(k,"x")
                    x_current[k,:] = xsol

                theta_current = x_current[:,4]
                theta_old = theta_current
                theta_diff = np.sum(np.abs(theta_current-theta_old))
        
            step_sol_x_arr = x_current
            velocity = step_sol_x_arr[0][3]
            steer = step_sol_x_arr[0][6]
            steer = np.rad2deg(steer)

            self.velocity_pub.publish(velocity)
            self.steer_pub.publish(steer)

            fig, axs = plt.subplots(3, 1, figsize=(5,10))
            axs[0].plot(self.path[0],self.path[1])
            axs[0].scatter(step_sol_x_arr[:,0], step_sol_x_arr[:,1],s=5)
            axs[1].plot(np.linspace(0,self.N,self.N),step_sol_x_arr[:,3])
            axs[2].plot(np.linspace(0,self.N,self.N),np.rad2deg(step_sol_x_arr[:,6]))
            plt.show()
            rate.sleep()

if __name__ == "__main__":
    rospy.init_node('controller_node')
    controller = MPCC()
    controller.control_loop()
