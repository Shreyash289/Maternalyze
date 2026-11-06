import numpy as np

# Categorical mappings
OBESE_MAP = {"No": 0, "Yes": 1}
ETHNICITY_MAP = {
    "GBR": 0, "OTH": 1, "WEU": 2, "ASE": 3, "CAR": 4, "CAE": 5, "MEA": 6,
    "NAF": 7, "IND": 8, "IRL": 9, "AFE": 10, "PAK": 11, "BGD": 12, "Unknown": 13,
}
DELIVERY_OUTCOME_MAP = {
    "Live Birth": 0, "Still Birth": 1, "Unknown": 2,
}
ONSET_OF_LABOUR_MAP = {
    "Spontaneous": 0, "Induced": 1, "Pre-labour Caesarean": 2, "Unknown": 3,
}
PRESENCE_OF_MECONIUM_MAP = {"No": 0, "Yes": 1, "Unknown": 2}
SHOULDER_DYSTOCIA_MAP = {"No": 0, "Yes": 1, "Unknown": 2}
SEX_MAP = {"Male": 0, "Female": 1, "Unknown": 2}
SEVERELY_PREMATURE_MAP = {"No": 0, "Yes": 1, "Unknown": 2}
GLUCOSE_TOLERANCE_MAP = {
    "Normal": 0,
    "Abnormal": 1,
    "Unknown": 2,
}
FOLIC_ACID_DOSE_MAP = {
    "Yes": 1,
    "No": 0,
    "Unknown": 2,
}
PRIMARY_INDICATION_MAP = {
    "Abnormal lie": 0,
    "Fetal anomalies": 1,
    "Failure to progress, first stage": 2,
    "Unknown": 3,
}
CATEGORY_CAESAREAN_MAP = {
    "Cat 1": 0,
    "Cat 2": 1,
    "Cat 3": 2,
    "Cat 4": 3,
    "Unknown": 4,
}
PERINEAL_CARE_MAP = {
    "Episiotomy": 0,
    "Second": 1,
    "Intact": 2,
    "First": 3,
    "Third": 4,
    "Other": 5,
    "Unknown": 6,
}
STILL_BIRTH_MAP = {
    "No": 0,
    "Yes": 1,
    "Unknown": 2,
}
MATERNITY_MONTH_MAP = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
    "Unknown": 0,
}


def preprocess_gdm_input(data):
    obese_val = OBESE_MAP.get(data.Obese, 0)
    ethnicity_val = ETHNICITY_MAP.get(data.Ethnicity, ETHNICITY_MAP["Unknown"])
    features = [
        data.AgeAtStartOfSpell or 0.0,
        data.WeightMeasured or 0.0,
        data.Height or 0.0,
        data.BodyMassIndexAtBooking or 0.0,
        obese_val,
        ethnicity_val,
        data.Glucoselevelblood or 0.0,
    ]
    return np.array(features).reshape(1, -1)


def preprocess_child_input(data):
    features = []
    features.append(data.Index_of_Multiple_Deprivation_Rank or 0.0)
    features.append(data.IMD_Decile or 0)
    features.append(data.AgeAtStartOfSpell or 0.0)
    features.append(data.WeightMeasured or 0.0)
    features.append(data.Height or 0.0)
    features.append(data.Body_Mass_Index_at_Booking or 0.0)
    features.append(OBESE_MAP.get(data.Obese, 0))
    features.append(ETHNICITY_MAP.get(data.Ethnicity, ETHNICITY_MAP["Unknown"]))
    features.append(0)  # Risk_Factors placeholder - extend as needed
    features.append(0)  # AntenatalMedicalFactors placeholder
    features.append(0)  # PreviousObstetricHistory placeholder
    features.append(data.Parity or 0)
    features.append(data.Gravida or 0)
    features.append(data.Glucoselevelblood or 0.0)
    features.append(GLUCOSE_TOLERANCE_MAP.get(data.GlucoseToleranceTest, GLUCOSE_TOLERANCE_MAP["Unknown"]))
    features.append(data.Glucoselevel0minblood or 0.0)
    features.append(data.Glucoselevel120minblood or 0.0)
    features.append(FOLIC_ACID_DOSE_MAP.get(data.FolicAcidDose, FOLIC_ACID_DOSE_MAP["Unknown"]))
    features.append(data.SystolicBloodPressureCuff or 0.0)
    features.append(data.Diastolic_Blood_Pressure or 0.0)
    features.append(data.VitaminDlevelblood or 0.0)
    features.append(data.O_Thyroidfunctionblood or 0.0)
    features.append(DELIVERY_OUTCOME_MAP.get(data.Delivery_Outcome, DELIVERY_OUTCOME_MAP["Unknown"]))
    features.append(ONSET_OF_LABOUR_MAP.get(data.OnsetofLabourMethod, ONSET_OF_LABOUR_MAP["Unknown"]))
    features.append(data.Contraction_frequency_prior_to_delivery or 0.0)
    features.append(PRIMARY_INDICATION_MAP.get(data.PrimaryIndicationforCaesarean, PRIMARY_INDICATION_MAP["Unknown"]))
    features.append(CATEGORY_CAESAREAN_MAP.get(data.Category_Caesarean_Section, CATEGORY_CAESAREAN_MAP["Unknown"]))
    features.append(PERINEAL_CARE_MAP.get(data.Perineal_care, PERINEAL_CARE_MAP["Unknown"]))
    features.append(data.EstimatedTotalBloodLoss or 0.0)
    features.append(data.Gestation or 0)
    features.append(SEVERELY_PREMATURE_MAP.get(data.Severely_Premature, SEVERELY_PREMATURE_MAP["Unknown"]))
    features.append(data.Gestation_Days or 0)
    features.append(data.Gestation_at_booking_Weeks or 0.0)
    features.append(data.No_Of_previous_Csections or 0)
    features.append(data.BabyBirthWeight or 0.0)
    features.append(PRESENCE_OF_MECONIUM_MAP.get(data.Presence_of_meconium, PRESENCE_OF_MECONIUM_MAP["Unknown"]))
    features.append(data.BW_Centile or 0.0)
    features.append(SHOULDER_DYSTOCIA_MAP.get(data.Shoulder_Dystocia, SHOULDER_DYSTOCIA_MAP["Unknown"]))
    features.append(data.LOS_mother_after_delivery or 0.0)
    features.append(SEX_MAP.get(data.Sex, SEX_MAP["Unknown"]))
    features.append(STILL_BIRTH_MAP.get(data.Still_Birth, STILL_BIRTH_MAP["Unknown"]))
    features.append(data.TotalApgarScoreat1minutes or 0)
    features.append(data.APGAR_Score_5 or 0)
    features.append(data.TotalApgarScoreat10minutes or 0)
    features.append(MATERNITY_MONTH_MAP.get(data.Maternity_Month, MATERNITY_MONTH_MAP["Unknown"]))

    return np.array(features).reshape(1, -1)


def get_gdm_precautions(prediction):
    if prediction == 1:
        return [
            "Maintain a healthy diet and exercise regularly.",
            "Monitor blood glucose levels frequently.",
            "Consult specialist for possible medication.",
            "Attend regular antenatal check-ups.",
        ]
    else:
        return ["Continue regular prenatal care."]


def get_child_precautions(prediction):
    if prediction == 1:
        return [
            "Increase monitoring of fetal development.",
            "Plan for possible neonatal care support.",
            "Prepare for potential premature delivery.",
            "Consult pediatric specialists as needed.",
        ]
    else:
        return ["Standard neonatal care recommended."]