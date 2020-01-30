$(document).ready(function () {
    const EXPERIMENT_API = "/demo/experiment/";
    const COMPARISON_API = "/demo/compare/";

    $('#querybox').autoComplete({
        resolverSettings: {
            url: '/demo/query_suggestion/?target_index='+$("#target_index").text()
        }
    });

    $('#gobtn').on('click', function(){
        const selected_configs = $(".config-row-checkbox:checked");
        if(selected_configs.length === 1) {
            const parent_form = selected_configs.closest("form");
            let query_params = "?" + parent_form.serialize();
            query_params += "&target_index=" + $("#target_index").val();
            window.location.href = EXPERIMENT_API+query_params;
        }
        else if(selected_configs.length === 2) {
            const parent_form_1 = $(selected_configs[0]).closest("form");
            const parent_form_2 = $(selected_configs[1]).closest("form");
            const form_1_params = parent_form_1.serializeArray();
            const form_2_params = parent_form_2.serializeArray();
            let config_object = {};

            /*
            Manually creating a url string in order to handle keys present in one config but not the other. If "gradkernels"
            is true for one config and the parameter "gradkernels" is not even applicable to the other config, the
            request params we should send to the server must be '?gradkernels=true&gradkernels=""'
             */
            $.map(form_1_params, function(item, i) {
                config_object[item.name] = [item.value, null];
            });

            $.map(form_2_params, function(item, i) {
                let key = item.name;
                if (!(key in config_object)){
                    config_object[key] = [null, item.value];
                }
                else {
                    config_object[key][1]= item.value;
                }
            });

            // let query_params = "?" + parent_form_1.serialize() + "&" + parent_form_2.serialize();
            let query_params = "?" + $.param(config_object, true);
            query_params += "&target_index=" + $("#target_index").val();
            window.location.href = COMPARISON_API+query_params;
        }
        else {
            alert("You must select at least 1 and utmost 2 config(s)");
        }
    });

    $("#comparison_table").DataTable({
        "order": [[1, "asc"]],
        "lengthChange": false,
        "pageLength": 25
    });

    $("body").on("keypress", function(e){
        if(e.which === 13) {
            let go_btn = $("#gobtn");
            if (go_btn.length) {
                go_btn.trigger("click");
            }
        }
    });

    $(".fake-table form").on("click", function(){
        let checkbox = $(this).find("input[type=checkbox]");
        checkbox.prop("checked", !checkbox.prop("checked"));
    });

    $("input[type=checkbox]").on("click", function(e){
        e.stopPropagation();
    });

});
