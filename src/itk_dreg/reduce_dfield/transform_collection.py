#!/usr/bin/env python3

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable

import itk
import numpy as np
import numpy.typing as npt

import itk_dreg.block.image as block_image

logger = logging.getLogger(__name__)


@dataclass
class TransformEntry:
    """
    Wraps a single transform for inclusion in a `TransformCollection` with or without a transform domain.
    """

    transform: itk.Transform
    """
    Transform representing some spatial relationship.
    """
    domain: Optional[itk.Image]
    """
    The domain over which the transform is valid.
    
    If a value is provided then the is considered to be bounded, meaning it is only valid to
    transform a point in real space by this transform if the transform falls within the
    physical bounds of the transform domain.
    As a convenience, a transform domain is constrained such that it must be representable
    with an oriented bounding box in physical space.
    An unbuffered `itk.Image` may be used to describe the oriented bounding box. The voxel grid
    subdivision of the `itk.Image` is ignored in `TransformCollection` processing.
    If `None` then the transform is considered to be unbounded, meaning it is valid to transform
    any point in real space by this transform to obtain another point in R^3.
    """


class TransformCollection:
    """
    Represent a collection of (possibly bounded) itk.Transform(s).

    A single point to transform may fall within multiple transform domains.
    In that case a simple average is taken of point output candidates.

    A domain of None indicates that a transform is valid over an
    unbounded, global input domain.

    TODO: Re-implement in ITK C++ to inherit from `itk.Transform`
        for use in filters and for improved performance.
    """

    @property
    def transforms(self) -> List[itk.Transform]:
        """The `itk.Transform`s in this collection."""
        return [entry.transform for entry in self.transform_and_domain_list]

    @property
    def domains(self) -> List[Optional[itk.Image]]:
        """The transform domains in this collection."""
        return [entry.domain for entry in self.transform_and_domain_list]

    @staticmethod
    def _bounds_contains(bounds: npt.ArrayLike, pt: npt.ArrayLike) -> bool:
        """
        Determines whether a point (X,Y,Z) falls within an axis-aligned physical bounding box.

        :param bounds: A 2x3 voxel array representing axis-aligned inclusive
            upper and lower bounds in (X,Y,Z) physical space.
        :param pt: A physical point (X,Y,Z).
        :return: True if the point is contained inside the inclusive physical region, else False.
        """
        return np.all(np.min(bounds, axis=0) <= pt) and np.all(
            np.max(bounds, axis=0) >= pt
        )

    @staticmethod
    def blend_simple_mean(
        input_pt: itk.Point, region_contributors: List[TransformEntry]
    ) -> npt.ArrayLike:
        """
        Method to blend among multiple possible transform outputs
        by performing unweighted averaging of all point candidates.

        If the input point falls within the domain overlap of three transforms,
        each transform will be applied independently to produce three point candidates
        and the output will be the linear sum of each candidate weighted by (1/3).

        This is a simple approach that may result in significant discontinuities
        at transform domain edges.

        :param input_pt: The point to transform.
        :param region_contributors: The transforms whose domains include the given point.
        :return: The average transformed point.
        """
        pts = [
            entry.transform.TransformPoint(input_pt) for entry in region_contributors
        ]
        return np.mean(pts, axis=0)

    @classmethod
    def blend_distance_weighted_mean(
        cls, input_pt: itk.Point, region_contributors: List[TransformEntry]
    ) -> npt.ArrayLike:
        """
        Blending method to weight transform results by their proximity
        to the edge of the corresponding transform domain.

        This blending approach avoids discontinuities at transform domain bounds.
        Transforms that have no bounds on the domain over which they apply are weighted minimally.

        #TODO Investigate alternatives to consider unbounded/background transform information.

        :param input_pt: The point to transform.
        :param region_contributors: The transforms whose domains include the given point.
        :return: The average transformed point.
        """

        MIN_WEIGHT = 1e-9
        point_candidates = [
            entry.transform.TransformPoint(input_pt) for entry in region_contributors
        ]
        weights = [
            (
                cls._physical_distance_from_edge(input_pt, entry.domain)[0]
                if entry.domain
                else MIN_WEIGHT
            )
            for entry in region_contributors
        ]

        if np.any([w < 0 for w in weights]):
            logger.error(
                "Detected at least one negative weight indicating"
                " a point unexpectedly lies outside a contributing region."
                " May impact transform blending results."
            )

        # Treat weights lying on an edge as if they were very small step inside the edge.
        # Domains are considered inclusive at bounds, meaning a single point candidate
        # at the boundary of a domain is a valid candidate and should be included in weighted averaging.
        interior_weights = [MIN_WEIGHT if np.isclose(w, 0) else w for w in weights]

        return np.average(point_candidates, axis=0, weights=interior_weights)

    @classmethod
    def _physical_distance_from_edge(
        cls, input_pt: itk.Point, domain: itk.Image
    ) -> Tuple[float, int]:
        """
        Estimate unsigned minimum physical distance to closest domain side.

        Handles domain with isotropic or anisotropic spacing over
        non-axis-aligned image domain representation.

        :param input_pt: The point to transform.
        :param domain: The transform domain to consider.
        :returns: Tuple with elements:
            0. The physical linear distance to the nearest image edge, and
            1. The zero-indexed axis to travel to reach the nearest edge.
        """
        # Set up unit vectors mapping from voxel to physical space
        voxel_step_vecs = np.matmul(
            np.array(domain.GetDirection()), np.eye(3) * itk.spacing(domain)
        )
        physical_step = np.ravel(
            np.take_along_axis(
                voxel_step_vecs,
                np.expand_dims(np.argmax(np.abs(voxel_step_vecs), axis=1), axis=1),
                axis=1,
            )
        )
        assert physical_step.ndim == 1 and physical_step.shape[0] == 3
        assert np.all(physical_step)

        pixel_axis_dists = cls._pixel_distance_from_edge(input_pt, domain)
        physical_axis_dists = [
            np.linalg.norm(axis_dist * physical_step)
            for axis_dist, physical_step in zip(pixel_axis_dists, physical_step)
        ]
        arg_min = np.argmin(np.abs(physical_axis_dists))
        return physical_axis_dists[arg_min], arg_min

    @staticmethod
    def _pixel_distance_from_edge(
        input_pt: itk.Point, domain: itk.Image
    ) -> npt.ArrayLike:
        """
        Estimate signed voxel distance to each image side.

        Inspired by
        https://github.com/InsightSoftwareConsortium/ITKMontage/blob/master/include/itkTileMergeImageFilter.hxx#L217

        :param input_pt: The point to transform.
        :param domain: The transform domain to consider.
        :return: The shortest pixel distance to an edge along each axis
            in ITK access order (I,J,K)
        """
        VOXEL_HALF_STEP = [0.5] * 3
        dist_to_lower_bound = np.array(
            domain.TransformPhysicalPointToContinuousIndex(input_pt)
        ) - (np.array(domain.GetLargestPossibleRegion().GetIndex()) - VOXEL_HALF_STEP)
        dist_to_upper_bound = (
            np.array(domain.GetLargestPossibleRegion().GetSize()) - dist_to_lower_bound
        )
        pixel_dists = np.array([dist_to_lower_bound, dist_to_upper_bound])
        axis_mins = np.ravel(
            np.take_along_axis(
                pixel_dists,
                np.expand_dims(np.argmin(np.abs(pixel_dists), axis=0), axis=0),
                axis=0,
            )
        )
        return axis_mins

    @staticmethod
    def _validate_entry(entry: TransformEntry) -> None:
        """
        Validate that an input transform entry is valid.

        :param entry: The bounded or unbounded transform entry.
        :raises TypeError: If either the transform or transform domain type is invalid.
        """
        if not issubclass(
            type(entry.transform), itk.Transform[itk.D, 3, 3]
        ) and not issubclass(type(entry.transform), itk.Transform[itk.F, 3, 3]):
            raise TypeError(f"Bad entry transform type: {type(entry.transform)}")

        if (
            entry.domain
            and itk.template(entry.domain)[0] != itk.Image
            and itk.template(entry.domain)[0] != itk.VectorImage
        ):
            raise TypeError(f"Bad entry domain type: {type(entry.domain)}")

    def __init__(
        self,
        transform_and_domain_list: List[TransformEntry] = None,
        blend_method: Callable[[itk.Point, List[TransformEntry]], itk.Point] = None,
    ):
        """
        Initialize a new `TransformCollection`.

        :param transform_and_domain_list: The list of transforms and associated transform domains
            to inform `TransformCollection` behavior.
        :param blend_method: The method to use to blend among output candidates in the case of
            overlapping transform domains.
        """
        transform_and_domain_list = transform_and_domain_list or []
        for entry in transform_and_domain_list:
            TransformCollection._validate_entry(entry)
        self.blend_method = (
            blend_method
            if blend_method
            else TransformCollection.blend_distance_weighted_mean
        )
        self.transform_and_domain_list = transform_and_domain_list

    def push(self, entry: TransformEntry) -> None:
        """
        Add a new bounded or unbounded transform to the underlying collection.

        :param entry: The transform and domain to add.
        :raises TypeError: If the entry is invalid.
        """
        TransformCollection._validate_entry(entry)
        self.transform_and_domain_list.append(entry)

    def transform_point(self, pt: itk.Point[itk.F, 3]) -> itk.Point[itk.F, 3]:
        """
        Transforms an input physical point (X,Y,Z) by the piecewise transform
        relationship developed by underlying bounded transforms and the
        selected blending method.

        :param pt: The physical point (X,Y,Z) to transform.
        :return: The transformed point (X,Y,Z) obtained after blending among
            point outputs from each viable transform candidate.
        :raises ValueError: If the input point does not fall within any
            of the transform domains contained within the `TransformCollection`.
        """
        region_contributors = [
            entry
            for entry in self.transform_and_domain_list
            if not entry.domain
            or TransformCollection._bounds_contains(
                block_image.get_sample_bounds(entry.domain), pt
            )
        ]
        if not region_contributors:
            raise ValueError(
                f"No candidates found: {pt} lies outside all transform domains"
            )
        return itk.Point[itk.F, 3](self.blend_method(pt, region_contributors))

    def TransformPoint(self, pt: itk.Point[itk.F, 3]) -> npt.ArrayLike:
        """
        `itk.Transform`-like interface to transform a point by the
        transform relationship developed by this `TransformCollection` instance.

        See `transform_point` documentation.
        """
        return self.transform_point(pt)
