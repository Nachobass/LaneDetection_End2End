"""
Author: Wouter Van Gansbeke
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
from Networks.utils import get_homography

class polynomial():
    """
    Polynomial class with exact integral calculation according to the
    trapezium rule.
    """
    def __init__(self, coeffs, a=0, b=0.7, n=100):
        self.a1, self.b1, self.c1 = torch.chunk(coeffs, 3, 1)
        self.a1, self.b1, self.c1 = self.a1.squeeze(), self.b1.squeeze(), self.c1.squeeze()
        self.a = a
        self.b = b
        self.n = n

    def calc_pol(self, x):
        return self.a1*x**2 + self.b1*x + self.c1

    def trapezoidal(self, other):
        h = float(self.b - self.a) / self.n
        s = 0.0
        s += abs(self.calc_pol(self.a)/2.0 - other.calc_pol(other.a)/2.0)
        for i in range(1, self.n):
            s += abs(self.calc_pol(self.a + i*h) - other.calc_pol(self.a + i*h))
        s += abs(self.calc_pol(self.b)/2.0 - other.calc_pol(self.b)/2.0)
        out = s*h
        return out


# pol1 = polynomial(coeffs=torch.FloatTensor([[0, 1, 0]]), a=-1, b=1)
# pol2 = polynomial(coeffs=torch.FloatTensor([[0, 0, 0]]), a=-1, b=1)
# pol1 = polynomial(coeffs=torch.FloatTensor([[0, 1, 0]]), a=0, b=1)
# pol2 = polynomial(coeffs=torch.FloatTensor([[1, 0, 0]]), a=0, b=1)
# print('Area by trapezium rule is {}'.format(pol1.trapezoidal(pol2)))


def define_loss_crit(options):
    '''
    Define loss cirterium:
        -MSE loss on curve parameters in ortho view
        -MSE loss on points after backprojection to normal view
        -Area loss
    '''
    if options.loss_policy == 'mse':
        loss_crit = MSE_Loss(options)
    elif options.loss_policy == 'homography_mse':
        loss_crit = Homography_MSE_Loss(options)
    elif options.loss_policy == 'backproject':
        loss_crit = backprojection_loss(options)
    elif options.loss_policy == 'area':
        loss_crit = Area_Loss(options.order, options.weight_funct, 
                              resize=options.resize, 
                              no_mapping=options.no_mapping,
                              no_cuda=options.no_cuda)
    else:
        return NotImplementedError('The requested loss criterion is not implemented')
    weights = torch.Tensor([1] + [options.weight_seg]*(options.nclasses))
    if not options.no_cuda:
        weights = weights.cuda()
    loss_seg = nn.CrossEntropyLoss(weights)
    # loss_seg = CrossEntropyLoss2d(seg=True, nclasses=options.nclasses, weight=options.weight_seg)
    return loss_crit, loss_seg


class CrossEntropyLoss2d(nn.Module):
    '''
    Standard 2d cross entropy loss on all pixels of image
    My implemetation (but since Pytorch 0.2.0 libs have their
    owm optimized implementation, consider using theirs)
    '''
    def __init__(self, weight=None, size_average=True, seg=False, nclasses=2, no_cuda=False):
        super(CrossEntropyLoss2d, self).__init__()
        if seg:
            weights = torch.Tensor([1] + [weight]*(nclasses))
            if not no_cuda:
                weights = weights.cuda()
            self.nll_loss = nn.NLLLoss2d(weights, size_average)
        else:
            self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets[:, 0, :, :])


class Area_Loss(nn.Module):
    '''
    Compute area between curves by integrating (x1 - x2)^2 over y
    *Area:
        *order 0: int((c1 - c2)**2)dy
        *order 1: int((b1*y - b2*y + c1 - c2)**2)dy
        *order 2: int((a1*y**2 - a2*y**2 + b1*y - b2*y + c1 - c2)**2)dy

    *A weight function W can be added:
        Weighted area: int(W(y)*diff**2)dy
        with W(y):
            *1
            *(1-y)
            *(1-y**0.5)
    '''
    def __init__(self, order, weight_funct, resize=256, no_mapping=False, no_cuda=False):
        super(Area_Loss, self).__init__()
        self.order = order
        self.weight_funct = weight_funct
        self.resize = resize
        self.no_mapping = no_mapping
        self.no_cuda = no_cuda
        
        # Get homography matrix for transforming coordinates to ortho view
        M, _ = get_homography(resize, no_mapping)
        # Store as double to match the dtype of coordinates (gt_params is double)
        self.M = torch.from_numpy(M).double()
        if not no_cuda:
            self.M = self.M.cuda()

    def _transform_to_ortho_view(self, x_coords, y_coords):
        """
        Transform coordinates from normal view to ortho (birds-eye) view using homography.
        Args:
            x_coords: [batch_size, num_points] x coordinates in normal view (resized)
            y_coords: [batch_size, num_points] y coordinates in normal view (resized)
        Returns:
            x_ortho: [batch_size, num_points] x coordinates in ortho view
            y_ortho: [batch_size, num_points] y coordinates in ortho view
        """
        batch_size = x_coords.size(0)
        num_points = x_coords.size(1)
        device = x_coords.device
        dtype = x_coords.dtype
        
        # Reshape for batch matrix multiplication
        # Stack coordinates as homogeneous coordinates: [batch_size, 3, num_points]
        ones = torch.ones(batch_size, num_points, dtype=dtype, device=device)
        coords = torch.stack([x_coords, y_coords, ones], dim=1)  # [batch_size, 3, num_points]
        
        # Transform using homography M (normal -> ortho)
        # M: [3, 3], coords: [batch_size, 3, num_points]
        M_batch = self.M.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, 3, 3]
        coords_ortho = torch.bmm(M_batch, coords)  # [batch_size, 3, num_points]
        
        # Normalize homogeneous coordinates
        x_ortho = coords_ortho[:, 0, :] / coords_ortho[:, 2, :]
        y_ortho = coords_ortho[:, 1, :] / coords_ortho[:, 2, :]
        
        return x_ortho, y_ortho

    def _fit_polynomial_coefficients(self, x_coords, y_coords, valid_mask):
        """
        Fit polynomial coefficients from x, y coordinates using least squares.
        Args:
            x_coords: [batch_size, num_points] x coordinates in ortho view
            y_coords: [batch_size, num_points] y coordinates in ortho view
            valid_mask: [batch_size, num_points] mask for valid points
        Returns:
            coeffs: [batch_size, order+1] polynomial coefficients
        """
        batch_size = x_coords.size(0)
        device = x_coords.device
        dtype = x_coords.dtype
        
        coeffs_list = []
        for i in range(batch_size):
            # Get valid points for this sample
            valid = valid_mask[i].bool()
            if valid.sum() < (self.order + 1):
                # Not enough points, return zero coefficients
                coeffs_list.append(torch.zeros(self.order + 1, dtype=dtype, device=device))
                continue
            
            x_valid = x_coords[i][valid]
            y_valid = y_coords[i][valid]
            
            # Apply the same y coordinate transformation as the model
            # Model uses: y_map = (255 - grid[:, :, 1]) where grid y is in ortho view
            # The grid is created with y in [0, H-1] where H=resize, so for resize=256: [0, 255]
            # After transformation to ortho view, grid y is still roughly in [0, resize-1] range
            # So y_map = (255 - y_ortho) for resize=256
            # The model hardcodes 255, so we'll use that for consistency
            y_map = 255.0 - y_valid  # Match model's y_map computation (hardcoded 255)
            # y_map is in [0, 255] range
            
            # Fit polynomial using y_map directly (not normalized) to match the model's coordinate system
            # The model fits: x = a*y_map^2 + b*y_map + c where y_map is in [0, 255]
            # We'll convert to normalized space later in the forward function
            if self.order == 0:
                Y = torch.ones(len(x_valid), 1, dtype=dtype, device=device)
            elif self.order == 1:
                Y = torch.stack([y_map, torch.ones(len(x_valid), dtype=dtype, device=device)], dim=1)
            elif self.order == 2:
                Y = torch.stack([y_map**2, y_map, torch.ones(len(x_valid), dtype=dtype, device=device)], dim=1)
            elif self.order == 3:
                Y = torch.stack([y_map**3, y_map**2, y_map, torch.ones(len(x_valid), dtype=dtype, device=device)], dim=1)
            else:
                raise NotImplementedError(f'Order {self.order} not implemented')
            
            # Solve least squares: Y * coeffs = x
            try:
                YtY = torch.mm(Y.t(), Y)
                # Add small regularization to avoid singular matrix
                reg = torch.eye(YtY.size(0), dtype=dtype, device=device) * 1e-6
                YtY_reg = YtY + reg
                Ytx = torch.mm(Y.t(), x_valid.unsqueeze(1))
                coeffs = torch.mm(torch.inverse(YtY_reg), Ytx).squeeze(1)
            except:
                # If matrix is singular, use zero coefficients
                coeffs = torch.zeros(self.order + 1, dtype=dtype, device=device)
            
            coeffs_list.append(coeffs)
        
        return torch.stack(coeffs_list, dim=0)

    def forward(self, params, gt_params, valid_samples=None):
        # params: [batch_size, order+1] or [batch_size, order+1, 1] - predicted polynomial coefficients
        params = params.squeeze(-1) if params.dim() > 2 else params
        
        # Check if gt_params are coordinates (size > order+1) or coefficients (size == order+1)
        if gt_params.size(1) > (self.order + 1):
            # gt_params are raw coordinates, need to fit polynomial coefficients
            # gt_params: [batch_size, num_points] x coordinates
            # valid_samples: [batch_size, num_points] mask for valid points
            
            if valid_samples is None:
                # Assume all non-negative/non-zero points are valid
                valid_samples = (gt_params > 0).double()
            
            # Get y coordinates (h_samples) - these should match the sampling used in dataloader
            # The y coordinates are fixed based on the dataset format
            num_points = gt_params.size(1)
            # h_samples from JSON are originally [240, 250, 260, ..., 710] (48 points)
            # The dataloader pads lanes to 56 points by adding -2 values at the beginning
            # So we need to pad h_samples to match (56 points total)
            # h_samples from JSON: [240, 250, ..., 710] (48 points)
            # After padding to 56: [dummy, ..., dummy, 240, 250, ..., 710]
            # In dataloader: h_samples = np.array(h_samples)/2.5 - 32 (after padding, but h_samples isn't padded in dataloader)
            # Actually, looking at the dataloader code, h_samples remains 48 points, but lanes are padded to 56
            # So the padded lanes have 8 extra -2 values at the beginning, but h_samples doesn't have corresponding values
            # We need to generate h_samples that match the padded lanes structure
            
            # Original h_samples: 48 points from 240 to 710
            start_h = 240
            delta = 10
            stop_h = 710 + delta  # 720, so np.arange(240, 720, 10) gives 48 points
            h_samples_original = np.arange(start_h, stop_h, delta)  # 48 points
            # Match the dataloader transformation: h_samples/2.5 - 32
            h_samples_resized = h_samples_original / 2.5 - 32  # 48 points
            
            # Now pad to match num_points (56)
            # The dataloader pads lanes with 8 -2 values at the beginning (56 - 48 = 8)
            # So we need to pad h_samples with 8 dummy values at the beginning
            pad_size = num_points - len(h_samples_resized)  # Should be 8
            if pad_size > 0:
                # Pad at the beginning with dummy values (they won't be used since valid_points will mask them)
                # Use the first valid h_sample value repeated for padding
                h_samples_padded = np.concatenate([np.full(pad_size, h_samples_resized[0]), h_samples_resized])
            else:
                h_samples_padded = h_samples_resized[:num_points]
            
            h_samples = torch.tensor(h_samples_padded, dtype=gt_params.dtype, device=gt_params.device)
            h_samples = h_samples.unsqueeze(0).expand(gt_params.size(0), -1)
            
            # Transform coordinates from normal view to ortho (birds-eye) view
            # This is necessary because the model outputs polynomial coefficients in ortho view
            x_ortho, y_ortho = self._transform_to_ortho_view(gt_params, h_samples)
            
            # Fit polynomial coefficients from ground truth coordinates in ortho view
            gt_coeffs = self._fit_polynomial_coefficients(x_ortho, y_ortho, valid_samples)
        else:
            # gt_params are already polynomial coefficients
            gt_coeffs = gt_params
        
        # The model outputs polynomial coefficients in terms of y_map (in [0, 255] range)
        # The Area_Loss integrates over y in [0, 0.7] normalized range (in [0, 1])
        # So we need to convert coefficients from y_map space to normalized y space
        # If x = a*y_map^2 + b*y_map + c, and y_map = 255*y_norm, then:
        # x = a*(255*y_norm)^2 + b*(255*y_norm) + c = a*255^2*y_norm^2 + b*255*y_norm + c
        # So the normalized coefficients are: (a*255^2, b*255, c)
        
        # Convert predicted coefficients from y_map space to normalized y space
        if self.order == 2:
            params_norm = torch.stack([
                params[:, 0] * (255.0 ** 2),  # a_norm = a * 255^2
                params[:, 1] * 255.0,          # b_norm = b * 255
                params[:, 2]                   # c_norm = c (unchanged)
            ], dim=1)
            gt_coeffs_norm = torch.stack([
                gt_coeffs[:, 0] * (255.0 ** 2),  # a_norm = a * 255^2
                gt_coeffs[:, 1] * 255.0,          # b_norm = b * 255
                gt_coeffs[:, 2]                   # c_norm = c (unchanged)
            ], dim=1)
        elif self.order == 1:
            params_norm = torch.stack([
                params[:, 0] * 255.0,  # b_norm = b * 255
                params[:, 1]           # c_norm = c (unchanged)
            ], dim=1)
            gt_coeffs_norm = torch.stack([
                gt_coeffs[:, 0] * 255.0,  # b_norm = b * 255
                gt_coeffs[:, 1]           # c_norm = c (unchanged)
            ], dim=1)
        else:
            # For order 0, no conversion needed
            params_norm = params
            gt_coeffs_norm = gt_coeffs
        
        # Compute difference between predicted and ground truth coefficients (in normalized space)
        diff = params_norm - gt_coeffs_norm
        a = diff[:, 0]
        b = diff[:, 1]
        t = 0.7 # up to which y location to integrate (normalized, in [0, 1])
        
        if self.order == 2:
            c = diff[:, 2]
            if self.weight_funct == 'none':
                # weight (1)
                loss_fit = (a**2)*(t**5)/5+2*a*b*(t**4)/4 + \
                           (b**2+c*2*a)*(t**3)/3+2*b*c*(t**2)/2+(c**2)*t
            elif self.weight_funct == 'linear':
                # weight (1-y)
                loss_fit = c**2*t - t**5*((2*a*b)/5 - a**2/5) + \
                           t**2*(b*c - c**2/2) - (a**2*t**6)/6 - \
                           t**4*(b**2/4 - (a*b)/2 + (a*c)/2) + \
                           t**3*(b**2/3 - (2*c*b)/3 + (2*a*c)/3)
            elif self.weight_funct == 'quadratic':
                # weight (1-y**0.5)
                loss_fit = t**3*(1/3*b**2 + 2/3*a*c) - \
                           t**(7/2)*(2/7*b**2 + 4/7*a*c) + \
                           c**2*t + 0.2*a**2*t**5 - 2/11*a**2*t**(11/2) - \
                           2/3*c**2*t**(3/2) + 0.5*a*b*t**4 - \
                           4/9*a*b*t**(9/2) + b*c*t**2 - 0.8*b*c*t**(5/2)
            else:
                return NotImplementedError('The requested weight function is \
                        not implemented, only order 1 or order 2 possible')
        elif self.order == 1:
            loss_fit = (b**2)*t + a*b*(t**2) + ((a**2)*(t**3))/3
        else:
            return NotImplementedError('The requested order is not implemented, only none, linear or quadratic possible')

        # Mask select if lane is present (check if there are valid points or non-zero coefficients)
        if valid_samples is not None:
            # Use valid_samples to determine if lane is present
            mask = (valid_samples.sum(dim=1) > 0).bool()
        else:
            # Fallback: check if coefficients are non-zero
            mask = (torch.abs(gt_coeffs).sum(dim=1) > 1e-6).bool()
        
        loss_fit = torch.masked_select(loss_fit, mask)
        loss_fit = loss_fit.mean(0) if loss_fit.size()[0] != 0 else torch.tensor(0.0, dtype=params.dtype, device=params.device)
        return loss_fit


class MSE_Loss(nn.Module):
    '''
    Compute mean square error loss on curve parameters
    in ortho or normal view
    '''
    def __init__(self, options):
        super(MSE_Loss, self).__init__()
        self.loss_crit = nn.MSELoss()
        if not options.no_cuda:
            self.loss_crit = self.loss_crit.cuda()

    def forward(self, params, gt_params, compute=True):
        loss = self.loss_crit(params.squeeze(-1), gt_params)
        return loss

class backprojection_loss(nn.Module):
    '''
    Compute mean square error loss on points in normal view
    instead of parameters in ortho view
    '''
    def __init__(self, options):
        super(backprojection_loss, self).__init__()
        M, M_inv = get_homography(options.resize, options.no_mapping)
        self.M, self.M_inv = torch.from_numpy(M), torch.from_numpy(M_inv)
        start = 160
        delta = 10
        num_heights = (720-start)//delta
        self.y_d = (torch.arange(start,720,delta)-80).double() / 2.5
        self.ones = torch.ones(num_heights).double()
        self.y_prime = (self.M[1,1:2]*self.y_d + self.M[1,2:])/(self.M[2,1:2]*self.y_d+self.M[2,2:])
        self.y_eval = 255 - self.y_prime

        if options.order == 0:
            self.Y = self.tensor_ones
        elif options.order == 1:
            self.Y = torch.stack((self.y_eval, self.ones), 1)
        elif options.order == 2:
            self.Y = torch.stack((self.y_eval**2, self.y_eval, self.ones), 1)
        elif options.order == 3:
            self.Y = torch.stack((self.y_eval**3, self.y_eval**2, self.y_eval, self.ones), 1)
        else:
            raise NotImplementedError(
                    'Requested order {} for polynomial fit is not implemented'.format(options.order))

        self.Y = self.Y.unsqueeze(0).repeat(options.batch_size, 1, 1)
        self.ones = torch.ones(options.batch_size, num_heights, 1).double()
        self.y_prime = self.y_prime.unsqueeze(0).repeat(options.batch_size, 1).unsqueeze(2)
        self.M_inv = self.M_inv.unsqueeze(0).repeat(options.batch_size, 1, 1)

        if not options.no_cuda:
            self.M = self.M.cuda()
            self.M_inv = self.M_inv.cuda()
            self.y_prime = self.y_prime.cuda()
            self.Y = self.Y.cuda()
            self.ones = self.ones.cuda()

    def forward(self, params, x_gt, valid_samples):
        # Sample at y_d in the homography space
        bs = params.size(0)
        x_prime = torch.bmm(self.Y[:bs], params)

        # Transform sampled points back
        coordinates = torch.stack((x_prime, self.y_prime[:bs], self.ones[:bs]), 2).squeeze(3).permute((0, 2, 1))
        trans = torch.bmm(self.M_inv[:bs], coordinates)
        x_cal = trans[:,0,:]/trans[:,2,:]
        # y_cal = trans[:,1,:]/trans[:,2,:] # sanity check

        # Compute error
        x_err = (x_gt-x_cal)*valid_samples
        loss = torch.sum(x_err**2) / (valid_samples.sum())
        if valid_samples.sum() == 0:
            loss = 0
        return loss, x_cal * valid_samples
