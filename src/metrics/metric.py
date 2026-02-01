import torch

def haversine_distance(pred,target):
    EARTH_RADIUS_M=6_371_000.0
    pred_lat,pred_lon =pred[:,0],pred[:,1]
    target_lat,target_lon =target[:,0],target[:,1]
    pred_lat=torch.deg2rad(pred_lat)
    pred_lon=torch.deg2rad(pred_lon)
    target_lat=torch.deg2rad(target_lat)
    target_lon=torch.deg2rad(target_lon)
    dlat=target_lat-pred_lat
    dlon=target_lon-pred_lon
    #here we implement the formula we took from wikipedia
    a=torch.sin(dlat/2)**2+\
        torch.cos(pred_lat)*torch.cos(target_lat)*torch.sin(dlon/2)**2
    c=2*torch.atan2(torch.sqrt(a),torch.sqrt(1-a))
    return EARTH_RADIUS_M*c
