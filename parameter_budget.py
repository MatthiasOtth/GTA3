# Parameters to tweak
# Set at most 1 parameter to -1
NUM_LAYERS = 16
HIDDEN_DIM = -1
OUT_DIM = 1
PARAM_BUDGET_CUST  = 1_000_000
# Parameters are independant of number of heads
# NUM_HEADS = 1

# Fixed params
IN_DIM = 29 # For zinc
PARAM_BUDGET_SMALL = 100_000
PARAM_BUDGET_LARGE = 500_000


def cost_attn_layer(ind, outd):
    QKV = ind * outd * 3
    return QKV

def cost_gta3_layer(ind, outd):
    QKV = cost_attn_layer(ind, outd)
    ff  = 5 * outd**2
    norm = 2* outd
    return QKV + ff + norm

def cost_gta3(ind, hid, outd, layers):
    embed = ind * hid
    gta3 = 0
    for i in range(layers-1):
        gta3 += cost_gta3_layer(hid, hid)
    gta3 += cost_gta3_layer(hid, outd)
    out_ff = 2 * outd**2 + outd*2
    total = embed + gta3 + out_ff

    return embed, gta3, out_ff, total

def calc_missing_layer():
    EMBED = IN_DIM * HIDDEN_DIM
    GTA3_BASE = cost_gta3_layer(HIDDEN_DIM, OUT_DIM)
    gta3_layer = cost_gta3_layer(HIDDEN_DIM, HIDDEN_DIM)
    OUT_FF = 2 * OUT_DIM**2 + OUT_DIM*2
    for budget in [PARAM_BUDGET_SMALL, PARAM_BUDGET_LARGE, PARAM_BUDGET_CUST]:
        layers = budget - EMBED - GTA3_BASE - OUT_FF
        layers = layers // gta3_layer

        embed, gta3, out_ff, total = cost_gta3(IN_DIM, HIDDEN_DIM, OUT_DIM, layers)
        print("===========Budget: ", budget, "===========")
        print("\tWith ", layers, " layers")
        print("\t\tEmbedding: ", embed)
        print("\t\tGTA3: ", gta3)
        print("\t\tOutput FF: ", out_ff)
        print("\t\tTotal: ", total)

        embed, gta3, out_ff, total = cost_gta3(IN_DIM, HIDDEN_DIM, OUT_DIM, layers+1)
        print("\tWith ", layers+1, " layers") 
        print("\t\tEmbedding: ", embed)
        print("\t\tGTA3: ", gta3)
        print("\t\tOutput FF: ", out_ff)
        print("\t\tTotal: ", total)

def calc_missing_hidden():
    hid = 0
    done = [False, False, False]
    while hid<1000:
        EMBED, GTA3, OUT_FF, total = cost_gta3(IN_DIM, hid, OUT_DIM, NUM_LAYERS)
        for i,budget in enumerate([PARAM_BUDGET_SMALL, PARAM_BUDGET_LARGE, PARAM_BUDGET_CUST]):
            if done[i]:
                continue
            if total > budget:
                done[i] = True
                embed, gta3, out_ff, total = cost_gta3(IN_DIM, hid-1, OUT_DIM, NUM_LAYERS)
                print("===========Budget: ", budget, "===========")
                print("\tWith ", hid-1, " hidden dim")
                print("\t\tEmbedding: ", embed)
                print("\t\tGTA3: ", gta3)
                print("\t\tOutput FF: ", out_ff)
                print("\t\tTotal: ", total)
                embed, gta3, out_ff, total = cost_gta3(IN_DIM, hid, OUT_DIM, NUM_LAYERS)
                print("\tWith ", hid, " hidden dim") 
                print("\t\tEmbedding: ", embed)
                print("\t\tGTA3: ", gta3)
                print("\t\tOutput FF: ", out_ff)
                print("\t\tTotal: ", total)

        hid += 1

def calc_missing_out():
    out = 0
    done = [False, False, False]
    while out<1000:
        EMBED, GTA3, OUT_FF, total = cost_gta3(IN_DIM, HIDDEN_DIM, out, NUM_LAYERS)
        for i,budget in enumerate([PARAM_BUDGET_SMALL, PARAM_BUDGET_LARGE, PARAM_BUDGET_CUST]):
            if done[i]:
                continue
            if total > budget:
                done[i] = True
                embed, gta3, out_ff, total = cost_gta3(IN_DIM, HIDDEN_DIM, out-1, NUM_LAYERS)
                print("===========Budget: ", budget, "===========")
                print("\tWith ", out-1, " out dim")
                print("\t\tEmbedding: ", embed)
                print("\t\tGTA3: ", gta3)
                print("\t\tOutput FF: ", out_ff)
                print("\t\tTotal: ", total)
                embed, gta3, out_ff, total = cost_gta3(IN_DIM, HIDDEN_DIM, out, NUM_LAYERS)
                print("\tWith ", out, " out dim") 
                print("\t\tEmbedding: ", embed)
                print("\t\tGTA3: ", gta3)
                print("\t\tOutput FF: ", out_ff)
                print("\t\tTotal: ", total)

        out += 1

def calc_cost():
    missing = [NUM_LAYERS<0, HIDDEN_DIM<0, OUT_DIM<0]
    if not any(missing):
        embed, gta3, out_ff, total = cost_gta3(IN_DIM, HIDDEN_DIM, OUT_DIM, NUM_LAYERS)
        print("Embedding: ", embed)
        print("GTA3: ", gta3)
        print("Output FF: ", out_ff)
        print("Total: ", total)
    elif NUM_LAYERS < 0:
        calc_missing_layer()
    elif HIDDEN_DIM < 0:
        calc_missing_hidden()
    elif OUT_DIM < 0:
        calc_missing_out()

calc_cost()