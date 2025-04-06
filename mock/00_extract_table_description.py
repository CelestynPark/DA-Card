import os
import pandas as pd

excel_path = 'table_description.xlsx'

output_dir = 'output/table_specs'
os.makedirs(output_dir, exist_ok=True)

xls = pd.ExcelFile(excel_path)

for sheet_name in xls.sheet_names:
    try:
        df = xls.parse(sheet_name)
        df = df.dropna(how='all').dropna(axis=1, how='all')

        safe_name = sheet_name.replace(' ', '_').replace('.', '').replace('/', '_')
        output_path = os.path.join(output_dir, f"{safe_name}.csv")

        df.to_csv(output_path, index=False)
        print(f"File saved: {output_path}")
    except Exception as e:
        print(f"Error : {sheet_name} , {e}")