def get_severity(size, my_dict):
    sev = 0
    #print(my_dict.keys())
    for rng in my_dict.keys():
        #print(rng)
        #print('size',size)
        if (size in rng):
            sev = my_dict[rng]
            break
    return(sev)

### Dictionaries cmr and echo measurements. 
#Source: https://academic.oup.com/ehjcimaging/article/20/12/1321/5572609

#### -------- LVEDI ---------- ####
# 'Opposite' values was considered normal
lvedvi_cmr_male_dict = {
            tuple(range(8,105)): "Normal",
            tuple(range(105,400)): "Abnormal"
}

lvedvi_cmr_male_severity_dict = {
            tuple(range(8,56)): "normal",
            tuple(range(56, 105)): "normal",
            tuple(range(105, 117)): "mildly abnormal",
            tuple(range(117, 129)): "moderately abnormal",
            tuple(range(129, 400)): "severely abnormal"
}

lvedvi_cmr_female_dict = {
            tuple(range(8, 96)): "Normal",
            tuple(range(96, 400)): "Abnormal"
}

lvedvi_cmr_female_severity_dict = {
            tuple(range(8, 56)): "normal",
            tuple(range(56, 96)): "normal",
            tuple(range(96, 106)): "mildly abnormal",
            tuple(range(106, 116)): "moderately abnormal",
            tuple(range(116, 400)): "severely abnormal"
}

lvedvi_echo_male_dict = {
            tuple(range(8,74)): "Normal",
            tuple(range(74,400)): "Abnormal"
}

lvedvi_echo_male_severity_dict = {
            tuple(range(8, 74)): "normal",
            tuple(range(74, 89)): "mildly abnormal",
            tuple(range(89, 100)): "moderately abnormal",
            tuple(range(100, 400)): "severely abnormal"
}

lvedvi_echo_female_dict = {
            tuple(range(8,61)): "Normal",
            tuple(range(61,400)): "Abnormal"
}

lvedvi_echo_female_severity_dict = {
            tuple(range(8, 61)): "normal",
            tuple(range(61, 70)): "mildly abnormal",
            tuple(range(70, 80)): "moderately abnormal",
            tuple(range(80, 400)): "severely abnormal"
}

#### -------- LVESI ---------- ####
# 'Opposite' values was considered normal
lvesvi_cmr_male_dict = {
            tuple(range(4,38)): "Normal",
            tuple(range(38,400)): "Abnormal"
}

lvesvi_cmr_male_severity_dict = {
            tuple(range(2, 38)): "normal",
            tuple(range(38, 44)): "mildly abnormal",
            tuple(range(44, 50)): "moderately abnormal",
            tuple(range(50, 400)): "severely abnormal"
}

lvesvi_cmr_female_dict = {
            tuple(range(2,34)): "Normal",
            tuple(range(34,400)): "Abnormal"
}

lvesvi_cmr_female_severity_dict = {
            tuple(range(2, 34)): "normal",
            tuple(range(34, 39)): "mildly abnormal",
            tuple(range(39, 44)): "moderately abnormal",
            tuple(range(44, 400)): "severely abnormal"
}

lvesvi_echo_male_dict = {
            tuple(range(2,31)): "Normal",
            tuple(range(31,400)): "Abnormal"
}

lvesvi_echo_male_severity_dict = {
            tuple(range(2, 31)): "normal",
            tuple(range(31, 38)): "mildly abnormal",
            tuple(range(38, 45)): "moderately abnormal",
            tuple(range(45, 400)): "severely abnormal"
}

lvesvi_echo_female_dict = {
            tuple(range(2,24)): "Normal",
            tuple(range(24,400)): "Abnormal"
}

lvesvi_echo_female_severity_dict = {
            tuple(range(2, 24)): "normal",
            tuple(range(24, 32)): "mildly abnormal",
            tuple(range(32, 40)): "moderately abnormal",
            tuple(range(40, 400)): "severely abnormal"
}

#### -------- LVEF ---------- ####

lvef_cmr_male_dict = {
            # tuple(range(77, 100)): "hyperdynamic",
            # tuple(range(57,77)): "normal",
            tuple(range(57, 100)): "normal",
            tuple(range(41,57)): "mildly abnormal",
            tuple(range(30,41)): "moderately abnormal",
            tuple(range(0,30)): "severely abnormal"
}

lvef_cmr_female_dict = {
            # tuple(range(77, 100)): "hyperdynamic",
            # tuple(range(57,77)): "normal",
            tuple(range(57, 100)): "normal",
            tuple(range(41,57)): "mildly abnormal",
            tuple(range(30,41)): "moderately abnormal",
            tuple(range(0,30)): "severely abnormal"
}

lvef_echo_male_dict = {
            # tuple(range(72,100)): "hyperdynamic",
            # tuple(range(52,72)): "normal",
            tuple(range(52,100)): "normal",
            tuple(range(41,52)): "mildly abnormal",
            tuple(range(30,41)): "moderately abnormal",
            tuple(range(0,30)): "severely abnormal"
}

lvef_echo_female_dict = {
            # tuple(range(74,100)): "hyperdynamic",
            # tuple(range(54,74)): "normal",
            tuple(range(54,100)): "normal",
            tuple(range(41,54)): "mildly abnormal",
            tuple(range(30,41)): "moderately abnormal",
            tuple(range(0,30)): "severely abnormal"
}

#### -------- LVMI ---------- ####

lvmi_cmr_male_severity_dict = {
            tuple(range(10,49)): "normal",
            tuple(range(49,86)): "normal",
            tuple(range(86,95)): "mildly abnormal",
            tuple(range(95,104)): "moderately abnormal",
            tuple(range(104,400)): "severely abnormal",
}
lvmi_cmr_male_dict = {
            tuple(range(10,86)): "Normal",
            tuple(range(86,400)): "Abnormal",
}

lvmi_cmr_female_severity_dict = {
            tuple(range(10,41)): "normal",
            tuple(range(41,82)): "normal",
            tuple(range(82,92)): "mildly abnormal",
            tuple(range(92,102)): "moderately abnormal",
            tuple(range(102,400)): "severely abnormal",
}
lvmi_cmr_female_dict = {
            tuple(range(10,82)): "Normal",
            tuple(range(82,400)): "Abnormal"
}

lvmi_echo_male_severity_dict = {
            tuple(range(10,50)): "normal",
            tuple(range(50,103)): "normal",
            tuple(range(103,117)): "mildly abnormal",
            tuple(range(117,131)): "moderately abnormal",
            tuple(range(131,400)): "severely abnormal",
}
lvmi_echo_male_dict = {
            tuple(range(10,103)): "Normal",
            tuple(range(103,400)): "Abnormal"
}

lvmi_echo_female_severity_dict = {
            tuple(range(10,44)): "normal",
            tuple(range(44,89)): "normal",
            tuple(range(89,101)): "mildly abnormal",
            tuple(range(101,113)): "moderately abnormal",
            tuple(range(113,400)): "severely abnormal",
}
lvmi_echo_female_dict = {
            tuple(range(10,89)): "Normal",
            tuple(range(89,400)): "Abnormal"
}


#### -------- LA ---------- ####
#source: 2015_chamberquantification ASE guidelines - Table 4 - pp. 10

lavi_echo_male_dict = { 
            tuple(range(4,35)): "Normal",
            tuple(range(35,400)): "Abnormal",
}

lavi_echo_male_severity_dict = { 
            tuple(range(4,35)): "normal",
            tuple(range(35,42)): "mildly abnormal",
            tuple(range(42,49)): "moderately abnormal",
            tuple(range(49,400)): "severely abnormal",
}

lavi_echo_female_dict = {
            tuple(range(4,35)): "Normal",
            tuple(range(35,400)): "Abnormal"
}

lavi_echo_female_severity_dict = {
            tuple(range(4,35)): "normal",
            tuple(range(35,42)): "mildly abnormal",
            tuple(range(42,49)): "moderately abnormal",
            tuple(range(49,400)): "severely abnormal",
}


#### -------- RA ---------- ####
#source: 2015_chamberquantification ASE guidelines - Table 13 - pp. 30
ravi_echo_male_dict = { 
            tuple(range(4,32)): "Normal",
            tuple(range(32,200)): "Abnormal",
}

ravi_echo_female_dict = {
            tuple(range(4,27)): "Normal",
            tuple(range(27,200)): "Abnormal",
}

#### -------- RV ---------- ####
#source: 2015_chamberquantification ASE guidelines - Table 8 - pp. 20

rveda_echo_male_dict = { 
            tuple(range(4,13)): "Normal",
            tuple(range(13,200)): "Abnormal",
}

rveda_echo_female_dict = {
            tuple(range(4,12)): "Normal",
            tuple(range(12,200)): "Abnormal",
}

rvesa_echo_male_dict = { 
            tuple(range(1,8)): "Normal",
            tuple(range(8,200)): "Abnormal",
}

rvesa_echo_female_dict = {
            tuple(range(1,7)): "Normal",
            tuple(range(7,200)): "Abnormal",
}

