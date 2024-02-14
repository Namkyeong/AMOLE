# For HIA
prompts=("Human intestinal absorbtion (HIA)" 
"The molecule is positive w.r.t. a property that is defined as 'the ability of the body to be absorbed from the human gastrointestinal system into the bloodstream of the human body'")

for dataset in HIA
do
    for prompt in "${prompts[@]}"
    do
        python evaluate_vs.py --input_model_config AMOLE.pth --dataset $dataset --prompt "$prompt" --device 0
    done
done

# For Pgp
prompts=("P-glycoprotein Inhibition" 
"This molecule is known to inhibit P-glycoprotein, which is an ABC transporter protein involved in intestinal absorption, drug metabolism, and brain penetration, and its inhibition can seriously alter a drug's bioavailability and safety")

for dataset in Pgp_Inhibition
do
    for prompt in "${prompts[@]}"
    do
        python evaluate_vs.py --input_model_config AMOLE.pth --dataset $dataset --prompt "$prompt" --device 0
    done
done

# For DILI
prompts=("Inducing liver injury"
"This molecule induces liver injury that is most commonly caused by Amoxicillin/clavulanate isoniazid, and nonsteroidal anti-inflammatory drugs")

for dataset in DILI
do
    for prompt in "${prompts[@]}"
    do
        python evaluate_vs.py --input_model_config AMOLE.pth --dataset $dataset --prompt "$prompt" --device 0
    done
done

# For VDR
prompts=("Vitamin D receptor (VDR)"
"This molecule is active w.r.t. Vitamin D receptor (VDR). The best pharmacophore hypothesis contains one hydrogen bond acceptor (A), one hydrogen bond donor (D) and two hydrophobic regions (H).")

for dataset in VDR
do
    for prompt in "${prompts[@]}"
    do
        python evaluate_vs.py --input_model_config AMOLE.pth --dataset $dataset --prompt "$prompt" --device 0
    done
done

# For Bioavailability
prompts=("Oral Bioavailability" 
"The molecule is positive w.r.t. a property that is defined as 'the rate and extent to which the active ingredient or active moiety is absorbed from a drug product and becomes available at the site of action'")

for dataset in Bioavailability
do
    for prompt in "${prompts[@]}"
    do
        python evaluate_vs.py --input_model_config AMOLE.pth --dataset $dataset --prompt "$prompt" --device 0
    done
done

# For BBB
prompts=("Blood-Brain Barrier penetration" 
"The molecule is able to penetrate the Blood-Brain Barrier, which is the protection layer that blocks most foreign drugs as a membrane separating circulating blood and brain extracellular fluid, to deliver to the site of action.")

for dataset in BBB
do
    for prompt in "${prompts[@]}"
    do
        python evaluate_vs.py --input_model_config AMOLE.pth --dataset $dataset --prompt "$prompt" --device 0
    done
done

# For hERG
prompts=("hERG Blocker" 
"This molecule blocks the hERG, which is crucial for the coordination of the heart's beating.")

for dataset in hERG
do
    for prompt in "${prompts[@]}"
    do
        python evaluate_vs.py --input_model_config AMOLE.pth --dataset $dataset --prompt "$prompt" --device 0
    done
done




# For HIV
prompts=("Active against HIV virus"
"This molecule is active against HIV virus. These drugs typically possess one or more of the following properties: Reverse Transcriptase Inhibitors, Protease Inhibitors, Integrase Inhibitors, Fusion Inhibitors, CCR5 Antagonists, Post-Attachment Inhibitors.")

for dataset in HIV
do
    for prompt in "${prompts[@]}"
    do 
        python evaluate_vs.py --input_model_config AMOLE.pth --dataset $dataset --prompt "$prompt" --device 0
    done
done
