"""Pure spending and VASP-convergence checks; compatible with cluster Python3.6."""
import xml.etree.ElementTree as ET

TERMINAL={'COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}


def budget_projection(jobs,observed,historical):
    actual=float(historical);remaining=0.0
    for job in jobs:
        row=observed.get(job['name'],{});spent=float(row.get('actual',0))
        actual+=spent
        if row.get('state','UNKNOWN').split()[0] not in TERMINAL:
            remaining+=max(0,float(job['cap'])-spent)
    return dict(actual=actual,remaining_caps=remaining,actual_plus_remaining_caps=actual+remaining)


def validate_output(xml_text,outcar_text,is_relax):
    root=ET.fromstring(xml_text)
    node=root.find('.//parameters//i[@name="NELM"]')
    nelm=int(node.text) if node is not None else 120
    calculations=root.findall('calculation')
    scf_steps=[len(c.findall('scstep')) for c in calculations]
    reached=outcar_text.count('aborting loop because EDIFF is reached')
    electronic=bool(calculations) and all(0<n<nelm for n in scf_steps) and reached>=len(calculations)
    ionic='reached required accuracy - stopping structural energy minimisation' in outcar_text
    return dict(valid=electronic and (not is_relax or ionic),electronic_converged=electronic,
                ionic_converged=ionic if is_relax else None,n_ionic_outputs=len(calculations),
                scf_steps=scf_steps,nelm=nelm,EDIFF_markers=reached)
