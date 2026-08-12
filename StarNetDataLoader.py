import torch, copy
import numpy as np
import torch.nn.functional as F

class StarnetDataLoader():
    def __init__(self, cfg):
        field_list      = cfg['starfind']['field_list'].split(',')
        self.field_list      = [str(f.strip()) for f in field_list]
        self.data_dim = cfg['starfind'].getint('region_dims')
        self.scaling = torch.load(cfg['starfind']['scale_file'], weights_only=False)
        self.sim_fields = copy.deepcopy(self.field_list)
        self.tmap = {'density':'Density', "H2_p0_density":"H2I_Density", "total_energy":"TotalEnergy"}  # maps parameter names to simulation fields
        for i, f in enumerate(self.sim_fields):
            if f in self.tmap.keys():
                self.sim_fields[i] = self.tmap[f]
        if 'velocity_divergence' in self.sim_fields:
            self.sim_fields.remove('velocity_divergence')
            self.sim_fields += ['x-velocity', 'y-velocity', 'z-velocity']
        if 'metals' in self.sim_fields:
            self.sim_fields.remove('metals')
            self.sim_fields += ['SN_Colour','Metal_Density']        


    def __call__(self, ds, grid_vol, regrid=False):
        """
            pseudo-dataloader for use with enzo data outputs being
            processed with yt.  Tranforms a ds.smoothed_covering_grid
            into a {1, N_field, dim, dim, dim} normalized torch Tensor
        """
        vol = torch.zeros((1,len(self.field_list), self.data_dim, self.data_dim, self.data_dim))
        d = self.data_dim
        for i,field in enumerate(self.field_list):
                    if field == 'velocity_divergence':
                        vs = ['%s-velocity'%s for s in 'xyz']
                        div = torch.zeros(d,d,d)
                        for ii, ax in enumerate(vs):
                            vol_field = grid_vol[ax]
                            if regrid:
                                org_sz = grid_vol['Density'].shape[-1]
                                vol_field = vol_field.view((1,1,org_sz, org_sz, org_sz))
                                vol_field = F.interpolate(vol_field, size=(d,d,d), mode='trilinear')
                                vol_field = vol_field.view(d,d,d)
                            div += torch.from_numpy(np.gradient(vol_field, axis=ii))
                        vol[0,i] = (div - self.scaling[field]['mean'].mean()) \
                                / self.scaling[field]['std'].mean()
                    # construct sum metal field
                    elif field == 'metals':
                        metals = grid_vol['Metal_Density']
                        sn_c = grid_vol['SN_Colour']
                        if regrid:
                                org_sz = grid_vol['Density'].shape[-1]
                                metals = metals.view((1,1,org_sz, org_sz, org_sz))
                                metals = F.interpolate(metals, size=(d,d,d), mode='trilinear')
                                metals = metals.view(d,d,d)
                                sn_c = sn_c.view((1,1,org_sz, org_sz, org_sz))
                                sn_c = F.interpolate(sn_c, size=(d,d,d), mode='trilinear')
                                sn_c = sn_c.view(d,d,d)
                            
                        vol[0,i] = sn_c
                        vol[0,i] = vol[0,i]-self.scaling['SN_Colour']['mean'].mean() \
                                        *(self.scaling['Metal_Density']['std'].mean())
                        vol[0,i] = vol[0,i] + (metals\
                                    -self.scaling['Metal_Density']['mean'].mean())\
                                    * self.scaling['SN_Colour']['std'].mean()
                        vol[0,i] = vol[0,i] / (self.scaling['SN_Colour']['std'].mean()*self.scaling['Metal_Density']['std'].mean())
                    # elif field == "H2_p0_density":
                    #     vol[0,i] = (grid_vol['density']*1e-3-self.scaling['H2_p0_density']['mean'].mean())/self.scaling['H2_p0_density']['std'].mean()
                    else:
                        vol_field = grid_vol[self.tmap[field]]
                        if regrid:
                                org_sz = grid_vol['Density'].shape[-1]
                                vol_field = vol_field.view((1,1,org_sz, org_sz, org_sz))
                                vol_field = F.interpolate(vol_field, size=(d,d,d), mode='trilinear')
                                vol_field = vol_field.view(d,d,d)

                        vol[0,i] = vol_field
                        vol[0,i] = (vol[0,i]-self.scaling[field]['mean'].mean()) \
                                        /(self.scaling[field]['std'].mean())
        return vol