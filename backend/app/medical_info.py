"""
Medical information and recommendations for Diabetic Retinopathy stages
"""

DR_MEDICAL_INFO = {
    0: {  # No DR
        "severity": "No Diabetic Retinopathy",
        "description": "No signs of diabetic retinopathy detected in the retinal image.",
        "risk_level": "Low",
        "current_state": {
            "vision_impact": "No vision impairment from diabetic retinopathy",
            "retinal_changes": "No visible retinal changes or abnormalities detected",
            "blood_vessel_status": "Blood vessels appear healthy and normal"
        },
        "recommendations": [
            "Continue annual eye examinations (free through NZ diabetes annual review)",
            "Maintain good blood sugar control (HbA1c < 7%)",
            "Monitor blood pressure regularly at your GP or pharmacy",
            "Maintain healthy lifestyle with regular exercise (consider Green Prescription)",
            "Follow diabetes management plan through your DHB diabetes service"
        ],
        "follow_up": "12 months",
        "urgency": "Routine",
        "risk_factors": {
            "progression_risk": "5-10% annual risk if diabetes is poorly controlled",
            "key_factors": [
                "Duration of diabetes",
                "Blood sugar control level",
                "Blood pressure",
                "Cholesterol levels"
            ]
        },
        "prevention_tips": [
            "Keep blood glucose levels within target range",
            "Take prescribed diabetes medications regularly",
            "Maintain healthy diet low in sugar and processed foods",
            "Exercise at least 150 minutes per week",
            "Don't smoke or quit smoking"
        ],
        "statistics": {
            "progression_rate": "5-10% per year may develop mild DR",
            "vision_loss_risk": "Very low with proper management",
            "reversibility": "Maintain this stage with good diabetes control"
        }
    },
    
    1: {  # Mild DR
        "severity": "Mild Non-Proliferative Diabetic Retinopathy",
        "description": "Early stage diabetic retinopathy with microaneurysms detected.",
        "risk_level": "Low-Moderate",
        "current_state": {
            "vision_impact": "Usually no symptoms or vision changes",
            "retinal_changes": "Small balloon-like swellings in retinal blood vessels (microaneurysms)",
            "blood_vessel_status": "Minor blood vessel weakening beginning"
        },
        "recommendations": [
            "Schedule eye exam every 6-9 months (refer to local optometrist or DHB eye clinic)",
            "Improve blood sugar control urgently (target HbA1c < 7%)",
            "Control blood pressure (< 130/80 mmHg) - check at GP or local pharmacy",
            "Request referral to DHB diabetes specialist service",
            "Start or optimize cholesterol management through your GP"
        ],
        "follow_up": "6-9 months",
        "urgency": "Moderate - Schedule soon",
        "risk_factors": {
            "progression_risk": "15-25% may progress to moderate DR within 3 years",
            "key_factors": [
                "Poor blood sugar control",
                "High blood pressure",
                "High cholesterol",
                "Pregnancy",
                "Kidney disease"
            ]
        },
        "prevention_tips": [
            "Intensify diabetes management immediately",
            "Monitor blood glucose more frequently",
            "Consider continuous glucose monitoring (CGM)",
            "Reduce sodium intake for blood pressure control",
            "Take prescribed medications without missing doses"
        ],
        "statistics": {
            "progression_rate": "25% progress to moderate DR in 3 years",
            "vision_loss_risk": "Low if managed properly",
            "reversibility": "Can potentially stabilize or improve with excellent diabetes control"
        }
    },
    
    2: {  # Moderate DR
        "severity": "Moderate Non-Proliferative Diabetic Retinopathy",
        "description": "Moderate stage with blocked blood vessels and retinal swelling concerns.",
        "risk_level": "Moderate-High",
        "current_state": {
            "vision_impact": "May have mild blurry vision or floaters",
            "retinal_changes": "Blood vessel blockage, retinal hemorrhages, cotton-wool spots",
            "blood_vessel_status": "Significant blood vessel damage and leakage"
        },
        "recommendations": [
            "See eye specialist within 2-3 months (request urgent DHB ophthalmology referral)",
            "Urgent diabetes management review at DHB diabetes clinic",
            "May need laser treatment evaluation (available at major NZ hospitals)",
            "Check for macular edema at eye specialist",
            "Consider intravitreal injections if macular involvement (funded by PHARMAC)"
        ],
        "follow_up": "3-4 months",
        "urgency": "High - Schedule within weeks",
        "risk_factors": {
            "progression_risk": "30-50% risk of progression to severe DR within 1 year",
            "key_factors": [
                "Uncontrolled blood sugar",
                "Hypertension",
                "Diabetic kidney disease",
                "High lipid levels",
                "Long diabetes duration (>10 years)"
            ]
        },
        "prevention_tips": [
            "Immediate aggressive diabetes control needed",
            "Consider insulin therapy if not already prescribed",
            "Monitor for vision changes daily",
            "Avoid heavy lifting or straining",
            "Report any sudden vision changes immediately"
        ],
        "statistics": {
            "progression_rate": "50% progress to severe DR within 1 year if untreated",
            "vision_loss_risk": "Moderate - increasing risk of vision problems",
            "reversibility": "Progression can be slowed but damage often permanent"
        }
    },
    
    3: {  # Severe DR
        "severity": "Severe Non-Proliferative Diabetic Retinopathy",
        "description": "Advanced retinopathy with high risk of vision-threatening complications.",
        "risk_level": "High",
        "current_state": {
            "vision_impact": "Blurred vision, dark spots, difficulty seeing at night",
            "retinal_changes": "Extensive hemorrhages, severe blood vessel blockage, retinal ischemia",
            "blood_vessel_status": "Severe blood vessel damage with poor blood flow to retina"
        },
        "recommendations": [
            "URGENT referral to DHB ophthalmology service (request priority referral)",
            "Likely need for laser photocoagulation treatment (available at public hospitals)",
            "Monthly eye monitoring at hospital eye clinic",
            "Aggressive systemic disease management through DHB diabetes service",
            "Prepare for possible anti-VEGF injections (PHARMAC funded)"
        ],
        "follow_up": "1 month or as directed by specialist",
        "urgency": "URGENT - See specialist within days",
        "risk_factors": {
            "progression_risk": "50-75% risk of proliferative DR within 1 year",
            "key_factors": [
                "Very high risk of vision loss",
                "Risk of vitreous hemorrhage",
                "Risk of retinal detachment",
                "Macular edema common",
                "Neovascularization risk"
            ]
        },
        "prevention_tips": [
            "Immediate specialist intervention required",
            "Strict blood sugar control essential",
            "Blood pressure must be controlled",
            "Avoid activities that increase eye pressure",
            "Consider low vision aids if needed"
        ],
        "statistics": {
            "progression_rate": "75% develop proliferative DR within 1 year without treatment",
            "vision_loss_risk": "High - significant risk of severe vision loss",
            "reversibility": "Limited - focus on preventing further deterioration"
        }
    },
    
    4: {  # Proliferative DR
        "severity": "Proliferative Diabetic Retinopathy",
        "description": "Most advanced stage with new abnormal blood vessel growth threatening vision.",
        "risk_level": "Critical",
        "current_state": {
            "vision_impact": "Severe vision problems, possible partial vision loss",
            "retinal_changes": "New fragile blood vessels growing (neovascularization), scarring",
            "blood_vessel_status": "Abnormal new vessels prone to bleeding and retinal detachment"
        },
        "recommendations": [
            "EMERGENCY - Go to nearest hospital eye department or call 111",
            "Immediate laser surgery likely at Auckland/Wellington/Christchurch Eye Departments",
            "Anti-VEGF injections urgently needed (PHARMAC funded emergency treatment)",
            "Possible vitrectomy surgery at major NZ hospital",
            "Close monitoring through hospital ophthalmology service"
        ],
        "follow_up": "Immediate specialist care",
        "urgency": "EMERGENCY - See specialist immediately",
        "risk_factors": {
            "progression_risk": "90-95% immediate risk of severe vision loss or blindness without treatment",
            "key_factors": [
                "Vitreous hemorrhage risk",
                "Retinal detachment risk",
                "Neovascular glaucoma risk",
                "Macular edema",
                "Complete vision loss possible"
            ]
        },
        "prevention_tips": [
            "Immediate surgical intervention often required",
            "Avoid any activities that could increase bleeding risk",
            "Sleep with head elevated to reduce bleeding",
            "Report ANY vision changes immediately",
            "Consider vision rehabilitation services"
        ],
        "statistics": {
            "progression_rate": "Without treatment, 50% risk of severe vision loss within 2 years",
            "vision_loss_risk": "Very high - immediate intervention critical",
            "reversibility": "Damage largely irreversible but treatment can preserve remaining vision"
        }
    }
}

def get_comprehensive_analysis(prediction_idx, confidence):
    """
    Get comprehensive medical analysis based on DR prediction
    """
    info = DR_MEDICAL_INFO.get(prediction_idx, DR_MEDICAL_INFO[0])
    
    # Add confidence-based modifier
    confidence_modifier = ""
    if confidence < 0.7:
        confidence_modifier = "Note: Confidence is moderate. Consider retaking image or getting second opinion."
    elif confidence < 0.5:
        confidence_modifier = "Warning: Low confidence in prediction. Strongly recommend clinical verification."
    
    # Create comprehensive response
    comprehensive_info = {
        **info,
        "confidence_note": confidence_modifier,
        "general_advice": {
            "lifestyle": [
                "Maintain healthy weight (BMI < 25)",
                "Exercise regularly (30 min/day)",
                "Follow diabetic diet strictly",
                "Monitor blood sugar daily",
                "Take all medications as prescribed"
            ],
            "monitoring": [
                "Regular HbA1c tests (every 3 months)",
                "Blood pressure checks",
                "Cholesterol monitoring",
                "Kidney function tests",
                "Regular foot examinations"
            ],
            "warning_signs": [
                "Sudden vision changes or loss",
                "Flashes of light in vision",
                "Dark curtain over vision",
                "Sudden increase in floaters",
                "Eye pain or pressure"
            ]
        },
        "resources": {
            "hotlines": [
                "Healthline NZ: 0800 611 116 (24/7 free health advice)",
                "Diabetes NZ: 0800 DIABETES (0800 342 238)",
                "Blind Low Vision NZ: 0800 24 33 33",
                "Emergency: 111 (for urgent eye emergencies)"
            ],
            "websites": [
                "diabetes.org.nz",
                "health.govt.nz/your-health/conditions-and-treatments/diseases-and-illnesses/diabetes",
                "blindlowvision.org.nz",
                "eyehealthaotearoa.org.nz",
                "maculardegenerationnz.org.nz"
            ],
            "nz_services": [
                "Free annual diabetes checks through your GP",
                "Retinal screening through local DHB services",
                "Green prescription for exercise support",
                "Community diabetes education programmes"
            ]
        }
    }
    
    return comprehensive_info