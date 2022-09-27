# -*- coding: utf-8 -*-
"""
Eamon K. Conway
Kostas Research Institute for Homeland Security
Northeastern University

This is an algorithm that orthorectifies images from a drone sensor. 
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, Parameter, printfuncs, fit_report
import pylas
from scipy.interpolate import RegularGridInterpolator,LinearNDInterpolator
from pyproj import Proj, transform, CRS
import rasterio
from scipy.optimize import fmin
import numba as nb
from tqdm import tqdm
#%matplotlib widget
from multiprocessing import Pool,Process
import time
from numba.typed import List
from sys import exit

#@nb.jit(fastmath=True)
def remap(lst):
    n = len(lst)
    m = len(lst[0])
    intersect_pts = np.zeros((n,m,3))
    idx_h = np.zeros((n,m))
    idx_w = np.zeros((n,m))
    for i in tqdm(range(n)):
        for j in range(m):
            idx_h[i,j] = lst[i][j][0]
            idx_w[i,j] = lst[i][j][1]
            intersect_pts[i,j,0] = lst[i][j][2]
            intersect_pts[i,j,1] = lst[i][j][3]
            intersect_pts[i,j,2] = lst[i][j][4]
    return(idx_h,idx_w,intersect_pts)

@nb.jit(fastmath=True)
def RotationMatrix(axis, theta):
    """
    This uses Euler-Rodrigues formula.
    axis = 3D vector
    theta = constant
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2)
    b, c, d = -axis * np.sin(theta / 2)
    a2, b2, c2, d2 = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([
        [a2 + b2 - c2 - d2, 2 * (bc - ad), 2 * (bd + ac)],
        [2 * (bc + ad), a2 + c2 - b2 - d2, 2 * (cd - ab)],
        [2 * (bd - ac), 2 * (cd + ab), a2 + d2 - b2 - c2]
    ])

@nb.jit(fastmath=True)
def rot_3D_ECEF(a1,a2,a3,a4,a5,a6,fov_h,fov_w,h,w,c1,o1,lon,lat):
    """
    Here we take in the first six angles in this order
    3 Mount: gamma (u), alpha (e'), beta (n')
    3 Drone: head (u), pitch (e'), roll (n')
    Then the FOV in y,x of image, plus number of y,x pixels
    Then C1 is a 3D vector, in ecef frame 
    Then lon/lat of C1
    Total rotation=R6*R5*R4*R3*R2*R1*R0’*R0*c1
    where R0 and R0' will be looped over
    """
    # define unit vectors in terms of ECEF coordinates
    unit_e = np.array([-np.sin(lon),np.cos(lon),0])
    unit_n = np.array([-np.cos(lon)*np.sin(lat),-np.sin(lon)*np.sin(lat),np.cos(lat)])
    unit_u = np.array([np.cos(lon)*np.cos(lat),np.sin(lon)*np.cos(lat),np.sin(lat)])
    

    #a1/R1 is for gamma mount about u
    R1 = RotationMatrix(unit_u,a1)
    #a2/R2 is for pitch mount about e'
    R2 = RotationMatrix( R1@unit_e ,a2)
    #a3/R3 is for roll mount about n'
    R3 = RotationMatrix( R2@R1@unit_n ,a3)
    
    # now for drone movement
    #a4 is for heading, ie movement about u
    R4 = RotationMatrix(unit_u,a4)
    # a5 is for pitch of drone about e'. 
    R5 = RotationMatrix( R4 @ unit_e ,a5)
    # a6 is for roll of drone about n'. 
    R6 = RotationMatrix( R5 @ R4 @ unit_n ,a6)
    
    
    step_h = fov_h/image_h
    step_w = fov_w/image_w
    
    image_coords = np.zeros((image_h,image_w,3))
    
    # the fov angles are derived and applied first to get the 
    for i in range(-int(image_h/2),int(image_h/2)):
        #angle h fov
        a7 = step_h*i
        R7 = RotationMatrix(unit_e,a7)
        for j in range(-int(image_w/2),int(image_w/2)):
            #angle w fov
            a8 = step_w*j
            R8 = RotationMatrix(unit_n, a8)
            image_coords[i+int(image_h/2),j+int(image_w/2),:] = o1 + R6@R5@R4@R3@R2@R1@R8@R7@c1
    return(image_coords)


class orthorectify_class(object):

    def __init__(self):
        pass

    def orthorectify(self,dr_alt_abv_surf,dr_lon,dr_lat,dr_lon_rad,dr_lat_rad,image_h,image_w,f,fov_h,fov_w,dem_file):
        self.dr_alt_abv_surf = dr_alt_abv_surf
        self.dr_lon = dr_lon
        self.dr_lat = dr_lat
        self.dr_lon_rad = dr_lon_rad
        self.dr_lat_rad = dr_lat_rad
        self.image_h = image_h
        self.image_w = image_w
        self.f = f
        self.fov_h = fov_h
        self.fov_w = fov_w
        self.dem_file = dem_file




        df = rasterio.open(self.dem_file)
        x = df.read()
        lon_min=df.bounds[0] ; lon_max=df.bounds[2] ; lat_min=df.bounds[1] ; lat_max=df.bounds[3]
        lon=np.zeros((x.shape[1],x.shape[2])) ; lat=np.zeros((x.shape[1],x.shape[2]))
        dem_lon = np.linspace(lon_min,lon_max,x.shape[1])
        dem_lat = np.linspace(lat_min,lat_max,x.shape[2])

        dem = x[0,::-1,:]
        #plt.imshow(dem[::-1,:])
        #plt.show()
        #exit()

        self.interp_dem = RegularGridInterpolator((dem_lat,dem_lon),dem)

        dr_alt_abv_dem = dr_alt_abv_surf + self.interp_dem((dr_lat,dr_lon))

        #dem to ECEF
        #inProj = CRS.from_string(str)
        outProj = Proj(init='epsg:4326')
        ecefProj = Proj(init='epsg:4978')

        dem_points = self.interp_dem((dem_lat,dem_lon)) 

        # Drone IMU Info
        dr_heading = 0#90*np.pi/180#90*np.pi/180  
        dr_roll = 0#10*np.pi/180#-10*np.pi/180
        dr_pitch = 10*np.pi/180#35*np.pi/180#10*np.pi/180

        #let us define a point directly below the drone (nadir direction) 
        cam_center_lon = dr_lon
        cam_center_lat = dr_lat
        cam_center_alt = dr_alt_abv_dem - 30

        self.cam_coords = np.array([cam_center_lon,cam_center_lat,dr_alt_abv_dem])
        cam_center_coords = np.array([cam_center_lon,cam_center_lat,cam_center_alt])


        # alpha mounting angle wrt pitch
        alpha = 0#35*np.pi/180#-90*np.pi/180 
        # beta mounting angle wrt to roll
        beta = 0#15*np.pi/180
        # gamma moutning angle wrt to yaw, but should not change central point
        gamma = 0#-45*np.pi/180

        a1,b1,c1 = transform(outProj,ecefProj,cam_center_coords[0],cam_center_coords[1],cam_center_coords[2])
        a2,b2,c2 = transform(outProj,ecefProj,self.cam_coords[0],self.cam_coords[1],self.cam_coords[2])
        d1=np.array([a1,b1,c1])
        d2=np.array([a2,b2,c2])
        l1 = d1-d2
        image_coords=rot_3D_ECEF(gamma,alpha,beta,dr_heading,dr_pitch,dr_roll,fov_h,fov_w,image_h,image_w,l1,d2,dr_lon_rad,dr_lat_rad)


        self.image_pts_lon = np.zeros((image_coords.shape[0],image_coords.shape[1]))
        self.image_pts_lat = np.zeros((image_coords.shape[0],image_coords.shape[1]))
        self.image_pts_alt = np.zeros((image_coords.shape[0],image_coords.shape[1]))


        for i in tqdm(range(image_coords.shape[0])):
            for j in range(image_coords.shape[1]):
                self.image_pts_lat[i,j],self.image_pts_lon[i,j],self.image_pts_alt[i,j]= self.ecef2lla_hugues(image_coords[i,j,0],image_coords[i,j,1],image_coords[i,j,2])

        self.image_pts_lat = self.image_pts_lat[::-1,:] 
        self.image_pts_lon = self.image_pts_lon[::-1,:] 
        self.image_pts_alt = self.image_pts_alt[::-1,:] 



        #"""
        #plt.plot(image_coords[0,:,2])
        #plt.plot(l2)
        #plt.show()
        fig,ax = plt.subplots(1,3,figsize=(12,12))
        fig.subplots_adjust(hspace=100)
        sc1=ax[0].imshow(self.image_pts_lon[::-1,:])
        fig.colorbar(sc1, ax=ax[0],shrink=0.3)
        sc2=ax[1].imshow(self.image_pts_lat[::-1,:])
        fig.colorbar(sc2, ax=ax[1],shrink=0.3)
        sc3=ax[2].imshow(self.image_pts_alt[::-1,:])
        fig.colorbar(sc3, ax=ax[2],shrink=0.3)
        ax[0].set_title(r'Lon [$^{o}$]')
        ax[1].set_title(r'Lat [$^{o}$]')
        ax[2].set_title('Alt [m]')
        plt.show()
        #"""
        #exit()
        tz = time.time()
        pool = Pool()
        results = pool.map(self.mp_get_intercepts,range(0,image_h))
        pool.close()
        pool.join()
        print('time to rectify = ',time.time()-tz)

        idx_h,idx_w,intersect_pts = remap(results)

        # now we have the central intersection point of our 2D area, let's get the corners of it
        # the full rotation consists of the mounted angles then the directional angles
        min_lon = np.min(intersect_pts[:,:,0])
        max_lon = np.max(intersect_pts[:,:,0])
        min_lat = np.min(intersect_pts[:,:,1])
        max_lat = np.max(intersect_pts[:,:,1])
        fig=plt.figure()
        sc=plt.imshow(intersect_pts[::-1,:,2],extent = [min_lon,max_lon,min_lat,max_lat],aspect='auto')
        plt.colorbar(sc)
        plt.show()
        plt.close()

        idx1 = np.logical_and(dem_lon<=max_lon,dem_lon>=min_lon)
        idx2 = np.logical_and(dem_lat<=max_lat,dem_lat>=min_lat)

        plt_dem = dem
        plt_dem = plt_dem[idx2,:]
        plt_dem = plt_dem[:,idx1]

        fig=plt.figure()
        plt.imshow(plt_dem,origin='lower',extent=[np.min(dem_lon[idx1]),np.max(dem_lon[idx1]),np.min(dem_lat[idx2]),np.max(dem_lat[idx2])],aspect='auto')
        plt.show()

 

    def ecef2lla_hugues(self,x, y, z):
        # x, y and z are scalars in meters (CANNOT use vectors for this method)
        # Following "An analytical method to transform geocentric into geodetic coordinates"
        # By Hugues Vermeille (2011)

        x = np.array([x]).reshape(np.array([x]).shape[-1], 1)
        y = np.array([y]).reshape(np.array([y]).shape[-1], 1)
        z = np.array([z]).reshape(np.array([z]).shape[-1], 1)

        a=6378137
        a_sq=a**2
        e = 8.181919084261345e-2
        e_sq = 6.69437999014e-3

        p = (x**2 + y**2)/a_sq
        q = ((1 - e_sq)*(z**2))/a_sq
        r = (p + q - e_sq**2)/6.

        evolute = 8*r**3 + p*q*(e_sq**2)

        if(evolute > 0):
                u = r + 0.5*(np.sqrt(8*r**3 + p*q*e_sq**2) + np.sqrt(p*q*e_sq**2))**(2/3.) + \
                0.5*(np.sqrt(8*r**3 + p*q*e_sq**2) - np.sqrt(p*q*e_sq**2))**(2/3.)
        else:
                u_term1 = np.sqrt(p*q*e_sq**2)/(np.sqrt(-8*r**3 - p*q*e_sq**2) + np.sqrt(-8*r**3))
                u_term2 = (-4.*r)*np.sin((2./3.)*np.arctan(u_term1))
                u_term3 = np.cos(np.pi/6. + (2./3.)*np.arctan(u_term1))
                u       = u_term2*u_term3

        v = np.sqrt(u**2 + q*e_sq**2)
        w = e_sq*(u + v - q)/(2.*v)
        k = (u + v)/(np.sqrt(w**2 + u + v) + w)
        d = k*np.sqrt(x**2 + y**2)/(k + e_sq)
        h = np.sqrt(d**2 + z**2)*(k + e_sq - 1)/k
        phi = 2.*np.arctan(z/((np.sqrt(d**2 + z**2) + d)))

        if((q == 0) and (p <= e_sq**2)):
                h = -(a*np.sqrt(1 - e_sq)*np.sqrt(e_sq - p))/(e)
                phi1 = 2*np.arctan(np.sqrt(e_sq**2 - p)/(e*(np.sqrt(e_sq - p)) + np.sqrt(1 - e_sq)*np.sqrt(p)))
                phi2 = -phi1
                phi = (phi1, phi2)


        case1 = (np.sqrt(2) - 1)*np.sqrt(y**2) < np.sqrt(x**2 + y**2) + x
        case2 = np.sqrt(x**2 + y**2) + y < (np.sqrt(2) + 1)*np.sqrt(x**2)
        case3 = np.sqrt(x**2 + y**2) - y < (np.sqrt(2) + 1)*np.sqrt(x**2)

        if(case1):
                #print("case1")
                lambd = 2.*np.arctan(y/(np.sqrt(x**2 + y**2) + x))
                return phi*180/np.pi, lambd*180/np.pi, h
        if(case2):
                #print("case2")
                lambd = (-np.pi/2) - 2.*np.arctan(x/(np.sqrt(x**2 + y**2) - y))
                return phi*180/np.pi, lambd*180/np.pi, h
        if(case3):
                #print("case3")
                lambd = (np.pi/2) - 2.*np.arctan(x/(np.sqrt(x**2 + y**2) + y))
                return phi*180/np.pi, lambd*180/np.pi, h

        # find intercept of line derived from old camera center and the rotated new camera center
        # with the dem
        # (a) define the line that joins these points
    def line3D(self,t,x1,x2):
        x = t*(x2[0]-x1[0]) + x1[0]
        y = t*(x2[1]-x1[1]) + x1[1]
        z = t*(x2[2]-x1[2]) + x1[2]
        return np.array([x,y,z])

        #coords on line
    def mp_get_intercepts(self,i):
        intersect_pts = []#np.zeros((image_pts_lon.shape[0],image_pts_lon.shape[1],3))
        x1=self.cam_coords
        #for i in range(image_pts_lon.shape[0])):
        for j in range(self.image_pts_lon.shape[1]):
            #print(i,j)
            t=1
            x2=np.array([self.image_pts_lon[i,j],self.image_pts_lat[i,j],self.image_pts_alt[i,j]])
            t_opt = fmin(self.target_func, x0=1,args=(x1,x2),xtol=1e-2,ftol=1e-1,disp=False)
            intersection_point = self.line3D(t_opt,x1,x2)
            pts = np.array([i,j],dtype=int)
            pts = np.concatenate((pts,intersection_point[:,0]))
            intersect_pts.append(pts)
        return(intersect_pts)


    def target_func(self,t,x1,x2):
        """Function that will be minimized by fmin
                :param t:      curve parameter of the straight line

                :returns:      (z_line(t) - z_surface(t))**2 – this is zero
                            at intersection points"""
        x = self.line3D(t,x1,x2)
        z = self.interp_dem((x[1],x[0]))
        res = np.sum((x[2] - z)**2)
        return res


if __name__ == '__main__':

    dr_alt_abv_surf = 122210
    dr_lon = -70.647463
    dr_lat = 41.534323
    dr_lon_rad = -70.647463*3.14/180
    dr_lat_rad = 41.534323*3.14/180
    image_w = 50#4352
    image_h = 50#3264
    f = 18 #mm
    fov_w = 15*3.14/180#*np.pi/180 #
    fov_h = 15*3.14/180#*np.pi/180 #

    dem_file = os.path.join(os.getcwd(),'n41_w071_1arc_v3.tif')

    x = orthorectify_class()
    x.orthorectify(dr_alt_abv_surf,dr_lon,dr_lat,dr_lon_rad,dr_lat_rad,image_h,image_w,f,fov_h,fov_w,dem_file)