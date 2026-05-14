using Pkg
Pkg.activate(".")
import JSON

function run_cells()
    path = "c:\\Users\\Ali-h\\OneDrive\\سطح المكتب\\practicum-5820-s2026-alajwadha\\CHEME-5820-Practicum-Student-S2026.ipynb"
    data = JSON.parsefile(path)
    ids = ["VSC-f02696ec", "VSC-d462df89", "VSC-3281b457", "VSC-48d96588", "VSC-5abab6fb", "VSC-75bcf010", "VSC-05feb929", "VSC-895a1806", "VSC-da5aa1ed", "VSC-5689cdea"]
    
    for id in ids
        println("Searching for cell: ", id)
        found = false
        for cell in data["cells"]
            source = join(cell["source"], "")
            if contains(source, id)
                println("Executing cell: ", id)
                if id in ["VSC-05feb929", "VSC-895a1806", "VSC-da5aa1ed"]
                    source = replace(source, r"(?m)^#\s*" => "")
                end
                try
                    include_string(Main, source)
                catch e
                    println("Error in cell ", id, ": ", e)
                end
                found = true
                break
            end
        end
        if !found
            println("Cell not found: ", id)
        end
    end
end

run_cells()
