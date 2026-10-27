"""Pure release check for the user's Claude03 conditional P1 authorization."""
def p1_gate(analysis,budget):
    if budget.get('approval_reference') in ['REVIEW_CLAUDE_04_L00.md','REVIEW_CLAUDE_05_fast_gpu.md']:
        return {'allowed':False,'reasons':['Latest review explicitly holds P1; old cost gate is superseded'],
                'baseline_actual_core_hours':{},'approval_reference':budget['approval_reference']}
    latest={}
    for row in analysis['attempts']:latest[row.get('logical_job',row['name'])]=row
    reasons=[];costs={}
    names=['03_L00_gamma','03b_L00_gamma_lrealauto','04_L01_gamma']
    for name in names:
        row=latest.get(name,{})
        acc=row.get('accounting',{})
        if acc.get('State')!='COMPLETED' or acc.get('ExitCode')!='0:0' or not row.get('vasp',{}).get('SCF_converged'):
            reasons.append(name+': awaiting a completed, independently checked SCF result')
        if name in ['03_L00_gamma','04_L01_gamma']:
            if not acc.get('CPUTimeRAW'):reasons.append(name+': missing allocation accounting')
            else:
                costs[name]=float(acc['CPUTimeRAW'])/3600
                if costs[name]>40:reasons.append(name+': actual allocation cost exceeds40core-hours')
    if budget['actual_plus_remaining_caps']>3000:reasons.append('total calibration reserve exceeds3000')
    if budget['p0_actual_plus_remaining_caps']>216:reasons.append('P0 actual plus reserve exceeds216')
    return {'allowed':not reasons,'reasons':reasons,'baseline_actual_core_hours':costs,
            'approval_reference':'REVIEW_CLAUDE_03_P0.md'}
