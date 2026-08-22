import pandas as pd

found = pd.read_csv("hospitals_geocoded.csv")
manual = pd.read_csv("hospitals_not_found.csv")

final = pd.concat(
    [found, manual],
    ignore_index=True
)

final.to_csv(
    "hospitals_final.csv",
    index=False
)

print("Successfully geocoded:", len(found))
print("Manually added:", len(manual))
print("Final total:", len(final))
