import DifferentiableCollisions as dc

function get_cbf_link_polygon_obstacle(obstacle)
    #=
        This function computes the value of
        α and ∂α/∂(r, q) between a polygon 
        obstacle and all of the robot links 
        that are relavant. This function requires 
        all of the links and obstacle be already 
        initialized.
    =#
    α2, _, J2 = dc.proximity_jacobian(obstacle, link2; verbose=false, pdip_tol=1e-6)
    α3, _, J3 = dc.proximity_jacobian(obstacle, link3; verbose=false, pdip_tol=1e-6)
    α4, _, J4 = dc.proximity_jacobian(obstacle, link4; verbose=false, pdip_tol=1e-6)
    α5, _, J5 = dc.proximity_jacobian(obstacle, link5; verbose=false, pdip_tol=1e-6)
    α6, _, J6 = dc.proximity_jacobian(obstacle, link6; verbose=false, pdip_tol=1e-6)
    α7, _, J7 = dc.proximity_jacobian(obstacle, link7; verbose=false, pdip_tol=1e-6)
    αhand, _, Jhand = dc.proximity_jacobian(obstacle, hand; verbose=false, pdip_tol=1e-6)

    return [α2, α3, α4, α5, α6, α7, αhand], [J2, J3, J4, J5, J6, J7, Jhand]
end