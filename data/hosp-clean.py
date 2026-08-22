import pandas as pd
a = pd.read_csv("unclean_hosp_chennai.csv")
a= a[((a["Facilities"]!="0")|(a["Specialties"]!="0"))&(a["District"]=="Chennai")]
a.to_csv("clean_hosp_chennai.csv",index=False)
