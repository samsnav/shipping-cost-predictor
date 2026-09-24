"""
Run the trained models against real, already-shipped pick tickets from
'Accuracy Test.xlsx' (sheet 'Static 9_17_2026') and compare predicted cost
to what was actually chosen, to sanity-check the models against real orders.

A pick ticket is made up of one or more lines (rows) sharing the same
pick_ticket_no; lines are combined into one shipment the same way
predict.predict_options_multi does for a caller-supplied multi-item list.

Usage:
    python accuracy_test.py
"""

import shutil
import tempfile

import pandas as pd

from predict import predict_options_multi

SOURCE_XLSX = 'Accuracy Test.xlsx'
SHEET_NAME = 'Static 9_17_2026'
OUTPUT_CSV = 'accuracy_test_predictions.csv'

# Keystone's own ship-from locations; every other ship_from_location_id in the
# export is a drop-ship vendor location, excluded here the same way training
# excludes them (see README: "Training is restricted to Keystone's own
# ship-from locations").
LOCATION_ID_TO_NAME = {
    100023: 'KT PA',
    194774: 'KT New KC',
    183806: 'KT PHX',
}

FREIGHT_MODES = {'LTL', 'FTL'}


def _actual_mode(carrier_mode: str) -> str:
    return 'LTL' if str(carrier_mode).upper().strip() in FREIGHT_MODES else 'PARCEL'


def load_static_tab(path=SOURCE_XLSX, sheet=SHEET_NAME) -> pd.DataFrame:
    # Copy to a temp file first so this works even when Excel has the file open
    tmp = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
    tmp.close()
    shutil.copy2(path, tmp.name)
    df = pd.read_excel(tmp.name, sheet_name=sheet)
    return df


def run():
    df = load_static_tab()
    df['location_name'] = df['ship_from_location_id'].map(LOCATION_ID_TO_NAME)

    n_tickets_total = df['pick_ticket_no'].nunique()
    excluded = df[df['location_name'].isna()]
    n_tickets_excluded = excluded['pick_ticket_no'].nunique()
    df = df.dropna(subset=['location_name'])
    n_tickets_used = df['pick_ticket_no'].nunique()
    print(f'{n_tickets_total} pick tickets total; '
          f'{n_tickets_excluded} excluded (drop-ship vendor locations); '
          f'{n_tickets_used} to predict.')

    rows = []
    for i, (ticket_no, group) in enumerate(df.groupby('pick_ticket_no'), start=1):
        first = group.iloc[0]
        items = list(zip(group['item_id'], group['ship_quantity']))
        actual_mode = _actual_mode(first['carrier_mode'])
        actual_tier = str(first['speed_tier'])

        try:
            options = predict_options_multi(
                first['location_name'], str(first['ship2_zip']), items
            )
        except Exception as exc:
            rows.append({
                'pick_ticket_no': ticket_no,
                'ship_from_location_name': first['location_name'],
                'ship_to_zip': first['ship2_zip'],
                'n_line_items': len(items),
                'total_qty': group['ship_quantity'].sum(),
                'actual_mode': actual_mode,
                'actual_speed_tier': actual_tier,
                'error': str(exc),
            })
            continue

        recommendation = options.pop('recommendation')
        row = {
            'pick_ticket_no': ticket_no,
            'ship_from_location_name': first['location_name'],
            'ship_to_zip': first['ship2_zip'],
            'n_line_items': len(items),
            'total_qty': group['ship_quantity'].sum(),
            'actual_mode': actual_mode,
            'actual_speed_tier': actual_tier,
            'recommendation_flag': recommendation['flag'],
        }
        for mode, tiers in options.items():
            if isinstance(tiers, str):
                row[f'{mode}_steer'] = tiers
            else:
                for tier, cost in tiers.items():
                    row[f'{mode}_{tier}'.replace(' ', '_')] = round(cost, 2)

        actual_col = f'{actual_mode}_{actual_tier}'.replace(' ', '_')
        row['predicted_cost_for_actual_choice'] = row.get(actual_col)

        rows.append(row)

        if i % 500 == 0:
            print(f'  {i}/{n_tickets_used} tickets predicted...')

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f'\nWrote {len(out)} predictions to {OUTPUT_CSV}')
    return out


if __name__ == '__main__':
    run()
