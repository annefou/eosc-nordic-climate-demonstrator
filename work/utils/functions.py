from imports import xr, xe, cftime, plt, ccrs, cm, cy, np, linregress, pd

import warnings


def rename_coords_lon_lat(ds):
    for k in ds.indexes.keys():
        if k == "longitude":
            ds = ds.rename({"longitude": "lon"})
        if k == "latitude":
            ds = ds.rename({"latitude": "lat"})

    return ds


def regrid_data(ds_in, ds_out):
    ds_in = rename_coords_lon_lat(ds_in)

    # Regridder data
    regridder = xe.Regridder(ds_in, ds_out,  "conservative")  #"bilinear")  #
    # regridder.clean_weight_file()

    # Apply regridder to data
    # the entire dataset can be processed at once
    ds_in_regrid = regridder(ds_in)

    # verify that the result is the same as regridding each variable one-by-one
    for k in ds_in.data_vars:
        print(k, ds_in_regrid[k].equals(regridder(ds_in[k])))

        if ds_in_regrid[k].equals(regridder(ds_in[k])) == True:
            ### Assign attributes from the original file to the regridded data
            #  ds_in_regrid.attrs['Conventions'] = ds_in.attrs['Conventions']
            # ds_in_regrid.attrs['history']     = ds_in.attrs['history']
            ds_in_regrid.attrs = ds_in.attrs

            ds_in_regrid[k].attrs["units"] = ds_in[k].attrs["units"]
            ds_in_regrid[k].attrs["long_name"] = ds_in[k].attrs["long_name"]
            try:
                ds_in_regrid[k].attrs["standard_name"] = ds_in[k].attrs["standard_name"]
                ds_in_regrid[k].attrs["comment"] = ds_in[k].attrs["comment"]
                ds_in_regrid[k].attrs["original_name"] = ds_in[k].attrs["original_name"]
                ds_in_regrid[k].attrs["cell_methods"] = ds_in[k].attrs["cell_methods"]
                ds_in_regrid[k].attrs["cell_measures"] = ds_in[k].attrs["cell_measures"]
            except KeyError:
                continue
    return ds_in_regrid


def plt_spatial_seasonal_mean(
    variable, vmin=None, vmax=None, levels=None, add_colorbar=None, title=None
):
    fig, axsm = plt.subplots(
        2, 2, figsize=[10, 7], subplot_kw={"projection": ccrs.PlateCarree()}
    )
    fig.suptitle(title, fontsize=16, fontweight="bold")
    axs = axsm.flatten()
    for ax, i in zip(axs, variable.season):
        im = variable.sel(season=i,).plot.contourf(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap=cm.devon_r,
            robust=True,
            vmin=vmin,
            vmax=vmax,
            levels=levels,
            extend="max",
            add_colorbar=add_colorbar,
        )

        ax.coastlines()
        gl = ax.gridlines()
        ax.add_feature(cy.feature.BORDERS)
        gl.top_labels = False
        ax.set_title("season: {}".format(i.values))

    plt.tight_layout()
    fig.subplots_adjust(top=0.88)

    return (
        fig,
        axs,
        im,
    )


def plt_diff_seasonal(
    true_var, estimated_var, cbar_label, vmin=None, vmax=None, levels=None, title=None
):

    fig, axsm = plt.subplots(
        2, 2, figsize=[10, 7], subplot_kw={"projection": ccrs.PlateCarree()}
    )
    fig.suptitle(title, fontsize=16, fontweight="bold")

    axs = axsm.flatten()
    for ax, i in zip(axs, true_var.season):
        im = (
            (true_var - estimated_var)
            .sel(season=i)
            .plot.contourf(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap=cm.bam,
                robust=True,
                add_colorbar=False,
                extend="both",
                vmin=vmin,
                vmax=vmax,
                levels=levels,
            )
        )
        # Plot cosmetics
        ax.coastlines()
        gl = ax.gridlines()
        ax.add_feature(cy.feature.BORDERS)
        gl.top_labels = False
        ax.set_title("season: {}".format(i.values))

    # colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([1, 0.15, 0.025, 0.7])
    cb = fig.colorbar(im, cax=cbar_ax, orientation="vertical", fraction=0.046, pad=0.04)
    # set cbar label
    cb.set_label(label=cbar_label, weight="bold")

    plt.tight_layout()
    fig.subplots_adjust(top=1)

    return axs


def plt_zonal_seasonal(variable_model, title=None, label=None):
    fig, axsm = plt.subplots(2, 2, figsize=[10, 7], sharex=True, sharey=True,)

    fig.suptitle(title, fontsize=16, fontweight="bold")

    axs = axsm.flatten()
    for ax, i in zip(axs, variable_model.season):
        for k, c in zip(
            variable_model.model.values,
            cm.romaO(range(0, 256, int(256 / len(variable_model.model.values)))),
        ):
            variable_model.sel(season=i, model=k).plot(
                ax=ax, label=k, color=c,
            )

        ax.set_ylabel(label, fontweight="bold")
        ax.grid()
        ax.set_title("season: {}".format(i.values))

    axs[1].legend(
        loc="upper left",
        bbox_to_anchor=(1.0, 1),
        title=label,
        fontsize="small",
        fancybox=True,
    )

    return axs


def plt_bar_area_mean(
    ax,
    var_model,
    var_obs,
    loc,
    bar_width=None,
    hatch=None,
    alpha=None,
    label=None,
    ylabel=None,
):

    for k, c, pos in zip(
        var_model.model.values,
        cm.romaO(range(0, 256, int(256 / len(var_model.model.values)))),
        range(len(var_model.model.values)),
    ):
        ax.bar(
            pos + loc * bar_width,
            var_model.sel(model=k).values,
            color=c,
            width=bar_width,
            edgecolor="black",
            hatch=hatch,
            alpha=alpha,
        )

    ax.bar(
        len(var_model.model.values) + loc * bar_width,
        var_obs.values,
        color="k",
        width=bar_width,
        edgecolor="white",
        hatch=hatch,
        alpha=alpha,
        label=label,
    )

    ax.set_xticks(range(len(np.append((var_model.model.values), "ERA5").tolist())))
    ax.set_xticklabels(
        np.append((var_model.model.values), "ERA5").tolist(), fontsize=12, rotation=90
    )
    ax.set_ylabel(ylabel, fontweight="bold")

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1, 1),
        title="MEAN",
        fontsize="small",
        fancybox=True,
    )

    plt.tight_layout()


def calc_regression(ds, ds_result, lat, step, season, model=None):

    x = ds["iwp_{}_{}".format(lat, lat + step)].sel(season=season).values.flatten()
    y = ds["sf_{}_{}".format(lat, lat + step)].sel(season=season).values.flatten()

    mask = ~np.isnan(y) & ~np.isnan(x)

    if x[mask].size == 0 or y[mask].size == 0:
        # print(
        #     "Sorry, nothing exists for {} in {}. ({}, {})".format(
        #         model, season, lat, lat + step
        #     )
        # )
        ds_result["slope_{}_{}".format(lat, lat + step,)] = np.nan
        ds_result["intercept_{}_{}".format(lat, lat + step,)] = np.nan
        ds_result["rvalue_{}_{}".format(lat, lat + step)] = np.nan
    else:
        _res = linregress(x[mask], y[mask])

        ds_result["slope_{}_{}".format(lat, lat + step,)] = _res.slope
        ds_result["intercept_{}_{}".format(lat, lat + step,)] = _res.intercept
        ds_result["rvalue_{}_{}".format(lat, lat + step)] = _res.rvalue

    return ds_result


def plt_scatter_iwp_sf_seasonal(
    ds, linreg, iteration, step, title=None, xlim=None, ylim=None
):

    fig, axsm = plt.subplots(2, 2, figsize=[10, 7], sharex=True, sharey=True,)
    fig.suptitle(title, fontsize=16, fontweight="bold")

    axs = axsm.flatten()
    for ax, i in zip(axs, ds.season):
        ax.grid()
        for _lat, c in zip(iteration, cm.romaO(range(0, 256, int(256 / 4)))):
            # plot scatter
            ax.scatter(
                ds["iwp_{}_{}".format(_lat, _lat + step)].sel(season=i),
                ds["sf_{}_{}".format(_lat, _lat + step)].sel(season=i),
                label="{}, {}".format(_lat, _lat + step),
                color=c,
                alpha=0.5,
            )

            # plot regression line
            y = (
                np.linspace(0, 350)
                * linreg["slope_{}_{}".format(_lat, _lat + step)].sel(season=i).values
                + linreg["intercept_{}_{}".format(_lat, _lat + step)]
                .sel(season=i)
                .values
            )

            ax.plot(np.linspace(0, 350), y, color=c, linewidth="2")

        ax.set_ylabel("Snowfall (mm$\,$day$^{-1}$)", fontweight="bold")
        ax.set_xlabel("Ice Water Path (g$\,$m$^{-2}$)", fontweight="bold")
        ax.set_title(
            "season: {}; lat: ({}, {})".format(
                i.values, iteration[0], iteration[-1] + step
            )
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    axs[1].legend(
        loc="upper left", bbox_to_anchor=(1, 1), fontsize="small", fancybox=True,
    )

    plt.tight_layout()


def return_ds_regression(ds, iteration, step):
    ds_result = dict()
    for season in ds.season.values:
        ds_res = xr.Dataset()
        for _lat in iteration:

            ds_res = calc_regression(ds, ds_res, _lat, step, season)

        ds_result[season] = ds_res

    ds_linear_reg = xr.concat(
        objs=list(ds_result.values()), dim=list(ds_result.keys()), coords="all"
    ).rename({"concat_dim": "season"})

    return ds_linear_reg


def return_pd_dataFrame_corr_coeff(ds, season, iteration, step):

    rvalue = []
    slope = []
    intercept = []
    index_season = []
    index_lat = []

    for lat in iteration:
        index_season.append("{}".format(season))
        index_lat.append("[{},{}]".format(lat, lat + step))
        rvalue.append(
            ds["rvalue_{}_{}".format(lat, lat + step)].sel(season=season).values.item()
        )
        slope.append(
            ds["slope_{}_{}".format(lat, lat + step)].sel(season=season).values.item()
        )
        intercept.append(
            ds["intercept_{}_{}".format(lat, lat + step)]
            .sel(season=season)
            .values.item()
        )

    da = pd.concat(
        [
            pd.Series(index_season),
            pd.Series(rvalue),
            pd.Series(slope),
            pd.Series(intercept),
        ],
        axis=1,
    )
    da.columns = ["season", "R**2", "a", "b"]
    return (da.T, pd.Series(index_lat))


# log_interpolate_1d_V2 is taken from the Metpy package https://github.com/Unidata/MetPy/blob/3dc05c8ae40a784956327907955fc09a3e46e0e1/src/metpy/interpolate/one_dimension.py#L53
def log_interpolate_1d_V2(x, xp, *args, axis=0, fill_value=np.nan):
    r"""Interpolates data with logarithmic x-scale over a specified axis.
    Interpolation on a logarithmic x-scale for interpolation values in pressure coordinates.
    Parameters
    ----------
    x : array-like
        1-D array of desired interpolated values.
    xp : array-like
        The x-coordinates of the data points.
    args : array-like
        The data to be interpolated. Can be multiple arguments, all must be the same shape as
        xp.
    axis : int, optional
        The axis to interpolate over. Defaults to 0.
    fill_value: float, optional
        Specify handling of interpolation points out of data bounds. If None, will return
        ValueError if points are out of bounds. Defaults to nan.
    Returns
    -------
    array-like
        Interpolated values for each point with coordinates sorted in ascending order.
    Examples
    --------
     >>> x_log = np.array([1e3, 1e4, 1e5, 1e6])
     >>> y_log = np.log(x_log) * 2 + 3
     >>> x_interp = np.array([5e3, 5e4, 5e5])
     >>> metpy.interpolate.log_interpolate_1d(x_interp, x_log, y_log)
     array([20.03438638, 24.63955657, 29.24472675])
    Notes
    -----
    xp and args must be the same shape.
    """
    # # Handle units
    # x, xp = _strip_matching_units(x, xp)

    # Log x and xp
    log_x = np.log(x)
    log_xp = np.log(xp)
    return interpolate_1d(log_x, log_xp, *args, axis=axis, fill_value=fill_value)


def interpolate_1d(x, xp, *args, axis=0, fill_value=np.nan, return_list_always=False):
    r"""Interpolates data with any shape over a specified axis.
    Interpolation over a specified axis for arrays of any shape.
    Parameters
    ----------
    x : array-like
        1-D array of desired interpolated values.
    xp : array-like
        The x-coordinates of the data points.
    args : array-like
        The data to be interpolated. Can be multiple arguments, all must be the same shape as
        xp.
    axis : int, optional
        The axis to interpolate over. Defaults to 0.
    fill_value: float, optional
        Specify handling of interpolation points out of data bounds. If None, will return
        ValueError if points are out of bounds. Defaults to nan.
    return_list_always: bool, optional
        Whether to always return a list of interpolated arrays, even when only a single
        array is passed to `args`. Defaults to ``False``.
    Returns
    -------
    array-like
        Interpolated values for each point with coordinates sorted in ascending order.
    Examples
    --------
     >>> x = np.array([1., 2., 3., 4.])
     >>> y = np.array([1., 2., 3., 4.])
     >>> x_interp = np.array([2.5, 3.5])
     >>> metpy.interpolate.interpolate_1d(x_interp, x, y)
     array([2.5, 3.5])
    Notes
    -----
    xp and args must be the same shape.
    """
    # # Handle units
    # x, xp = _strip_matching_units(x, xp)

    # Make x an array
    x = np.asanyarray(x).reshape(-1)

    # Save number of dimensions in xp
    ndim = xp.ndim

    # Sort input data
    sort_args = np.argsort(xp, axis=axis)
    sort_x = np.argsort(x)

    # indices for sorting
    sorter = broadcast_indices(xp, sort_args, ndim, axis)

    # sort xp
    xp = xp[sorter]
    # Ensure pressure in increasing order
    variables = [arr[sorter] for arr in args]

    # Make x broadcast with xp
    x_array = x[sort_x]
    expand = [np.newaxis] * ndim
    expand[axis] = slice(None)
    x_array = x_array[tuple(expand)]

    # Calculate value above interpolated value
    minv = np.apply_along_axis(np.searchsorted, axis, xp, x[sort_x])
    minv2 = np.copy(minv)

    # If fill_value is none and data is out of bounds, raise value error
    if ((np.max(minv) == xp.shape[axis]) or (np.min(minv) == 0)) and fill_value is None:
        raise ValueError("Interpolation point out of data bounds encountered")

    # Warn if interpolated values are outside data bounds, will make these the values
    # at end of data range.
    if np.max(minv) == xp.shape[axis]:
        warnings.warn("Interpolation point out of data bounds encountered")
        minv2[minv == xp.shape[axis]] = xp.shape[axis] - 1
    if np.min(minv) == 0:
        minv2[minv == 0] = 1

    # Get indices for broadcasting arrays
    above = broadcast_indices(xp, minv2, ndim, axis)
    below = broadcast_indices(xp, minv2 - 1, ndim, axis)

    if np.any(x_array < xp[below]):
        warnings.warn("Interpolation point out of data bounds encountered")

    # Create empty output list
    ret = []

    # Calculate interpolation for each variable
    for var in variables:
        # Var needs to be on the *left* of the multiply to ensure that if it's a pint
        # Quantity, it gets to control the operation--at least until we make sure
        # masked arrays and pint play together better. See https://github.com/hgrecco/pint#633
        var_interp = var[below] + (var[above] - var[below]) * (
            (x_array - xp[below]) / (xp[above] - xp[below])
        )

        # Set points out of bounds to fill value.
        var_interp[minv == xp.shape[axis]] = fill_value
        var_interp[x_array < xp[below]] = fill_value

        # Check for input points in decreasing order and return output to match.
        if x[0] > x[-1]:
            var_interp = np.swapaxes(np.swapaxes(var_interp, 0, axis)[::-1], 0, axis)
        # Output to list
        ret.append(var_interp)

    if return_list_always or len(ret) > 1:
        return ret
    else:
        return ret[0]


def broadcast_indices(x, minv, ndim, axis):
    """Calculate index values to properly broadcast index array within data array.
    See usage in interp.
    """
    ret = []
    for dim in range(ndim):
        if dim == axis:
            ret.append(minv)
        else:
            broadcast_slice = [np.newaxis] * ndim
            broadcast_slice[dim] = slice(None)
            dim_inds = np.arange(x.shape[dim])
            ret.append(dim_inds[tuple(broadcast_slice)])
    return tuple(ret)
