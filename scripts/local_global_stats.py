import csv, statistics as st

base = '/Users/ricardocarrillo/Desktop/Itau_UROP/itau-group4/results/baseline'
cols = ['mean_cos_sim_real_to_source', 'mean_cos_sim_fake_to_source',
        'intra_real_cohesion', 'intra_fake_cohesion', 'inter_group_separation',
        'silhouette_score', 'centroid_cosine_distance',
        'real_spread', 'fake_spread', 'cohens_d']

for enc in ['hubert', 'openl3', 'senet']:
    p = f'{base}/{enc}/metrics/experiment3_aggregate_analysis_{enc}.csv'
    rows = list(csv.DictReader(open(p)))
    print(f'===== {enc}  (n={len(rows)} timestamps)')
    for c in cols:
        v = [float(r[c]) for r in rows if r[c] not in ('', 'None')]
        if not v:
            continue
        print(f'  {c:30s} mean={st.mean(v):9.4f} med={st.median(v):9.4f} '
              f'sd={st.pstdev(v):8.4f} min={min(v):8.4f} max={max(v):8.4f}')
    gap = [float(r['mean_cos_sim_real_to_source']) - float(r['mean_cos_sim_fake_to_source'])
           for r in rows]
    print(f'  LOCAL GAP  mean={st.mean(gap):.4f} med={st.median(gap):.4f} '
          f'frac>0={sum(1 for g in gap if g > 0) / len(gap):.3f}')
    sil = [float(r['silhouette_score']) for r in rows if r['silhouette_score'] not in ('', 'None')]
    print(f'  silhouette>0 frac = {sum(1 for s in sil if s > 0) / len(sil):.3f}')
    cd = [float(r['cohens_d']) for r in rows if r['cohens_d'] not in ('', 'None')]
    print(f'  cohens_d>1 frac = {sum(1 for s in cd if s > 1) / len(cd):.3f}')
